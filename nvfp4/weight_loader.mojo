# ===----------------------------------------------------------------------=== #
# NVFP4 Weight Loader — Safetensors & GGUF for Qwen 3.5
#
# Loads NVFP4-quantized weights from:
#   1. Safetensors: 8-byte LE header_size + JSON metadata + raw tensor bytes
#   2. GGUF: MXFP4_MOE quantized weights (unsloth format)
#
# Converts to the packed format expected by nvfp4_gemm_gpu.mojo:
#   - packed_weights: UnsafePointer[UInt8]  [out, in//2]
#   - blockscales: UnsafePointer[Scalar[float8_e4m3fn]]  [out, in//16]
#   - weight_scale_2: Float32 (per-tensor second-level scale)
#   - input_scale: Float32 (activation calibration scale)
#
# Data is stored in List[UInt8] buffers. Extract UnsafePointer via
# .unsafe_ptr() when passing to GEMM kernels.
# ===----------------------------------------------------------------------=== #

from sys import external_call

from .config import (
    NVFP4_GROUP_SIZE,
    NVFP4_PACK_FACTOR,
    NvFp4Config,
    NvFp4Linear,
    pack_fp4_pair,
)


# ===----------------------------------------------------------------------=== #
# File I/O helpers (POSIX)
# ===----------------------------------------------------------------------=== #

comptime _O_RDONLY: Int = 0
comptime _SEEK_SET: Int = 0
comptime _SEEK_END: Int = 2


fn _fd_open(path: UnsafePointer[UInt8]) -> Int32:
    """Open file read-only. Returns fd or -1."""
    return external_call["open", Int32](path, Int32(_O_RDONLY))


fn _fd_close(fd: Int32):
    _ = external_call["close", Int32](fd)


fn _fd_read(fd: Int32, buf: UnsafePointer[UInt8], count: Int) -> Int:
    return Int(external_call["read", Int64](fd, buf, UInt64(count)))


fn _fd_lseek(fd: Int32, offset: Int64, whence: Int32) -> Int64:
    return external_call["lseek", Int64](fd, offset, whence)


fn _fd_file_size(fd: Int32) -> Int:
    """Get file size via lseek to end and back."""
    var end = _fd_lseek(fd, 0, Int32(_SEEK_END))
    _ = _fd_lseek(fd, 0, Int32(_SEEK_SET))
    return Int(end)


fn _fd_pread(fd: Int32, offset: Int, dst: UnsafePointer[UInt8], count: Int) -> Int:
    """Read count bytes at file offset into dst. Handles partial reads."""
    _ = _fd_lseek(fd, Int64(offset), Int32(_SEEK_SET))
    var total = 0
    while total < count:
        var n = _fd_read(fd, dst + total, count - total)
        if n <= 0:
            break
        total += n
    return total


fn _fd_read_into_list(fd: Int32, offset: Int, count: Int) -> List[UInt8]:
    """Read count bytes at file offset into a new List[UInt8]."""
    var buf = List[UInt8](capacity=count)
    for _ in range(count):
        buf.append(0)
    _ = _fd_pread(fd, offset, buf.unsafe_ptr(), count)
    return buf^


fn _str_to_cpath(s: String) -> List[UInt8]:
    """Convert String to null-terminated byte buffer for C APIs."""
    var buf = List[UInt8](capacity=len(s) + 1)
    var bytes = s.as_bytes()
    for i in range(len(s)):
        buf.append(bytes[i])
    buf.append(0)
    return buf^


fn _try_open_file(path: String) -> Int32:
    """Try to open a file. Returns fd >= 0 on success, -1 on failure."""
    var pbuf = _str_to_cpath(path)
    var fd = _fd_open(pbuf.unsafe_ptr())
    return fd


fn _get_byte(s: String, idx: Int) -> UInt8:
    """Get byte at index from String."""
    return s.as_bytes()[idx]


# ===----------------------------------------------------------------------=== #
# Minimal JSON field extraction (no full parser needed)
# ===----------------------------------------------------------------------=== #


fn _json_find_string(json: String, key: String) -> String:
    """Extract a JSON string value for the given key. Returns empty on miss."""
    var needle = '"' + key + '"'
    var pos = json.find(needle)
    if pos < 0:
        return String("")
    pos += len(needle)
    # Skip colon and whitespace.
    while pos < len(json):
        var b = _get_byte(json, pos)
        if b != ord(" ") and b != ord(":") and b != ord("\n") and b != ord("\t"):
            break
        pos += 1
    if pos >= len(json) or _get_byte(json, pos) != ord('"'):
        return String("")
    pos += 1  # skip opening quote
    var end = pos
    while end < len(json) and _get_byte(json, end) != ord('"'):
        end += 1
    return String(json[pos:end])


fn _json_find_int_array(json: String, key: String) -> List[Int]:
    """Extract a JSON integer array [n, m, ...]. Returns empty list on miss."""
    var result = List[Int]()
    var needle = '"' + key + '"'
    var pos = json.find(needle)
    if pos < 0:
        return result^
    pos += len(needle)
    # Find opening bracket.
    while pos < len(json) and _get_byte(json, pos) != ord("["):
        pos += 1
    if pos >= len(json):
        return result^
    pos += 1
    while pos < len(json) and _get_byte(json, pos) != ord("]"):
        # Skip whitespace and commas.
        var b = _get_byte(json, pos)
        if b == ord(" ") or b == ord(",") or b == ord("\n") or b == ord("\t"):
            pos += 1
            continue
        if b == ord("]"):
            break
        # Parse integer.
        var start = pos
        while pos < len(json):
            var c = _get_byte(json, pos)
            if c < ord("0") or c > ord("9"):
                break
            pos += 1
        if pos > start:
            var val = 0
            for i in range(start, pos):
                val = val * 10 + (Int(_get_byte(json, i)) - Int(ord("0")))
            result.append(val)
    return result^


# ===----------------------------------------------------------------------=== #
# Safetensors format
# ===----------------------------------------------------------------------=== #


struct TensorInfo(Copyable, Movable, ImplicitlyCopyable):
    """Metadata for one tensor inside a safetensors file."""
    var name: String
    var dtype: String        # "U8", "F8_E4M3", "F32", "BF16", etc.
    var shape: List[Int]     # e.g. [out_features, in_features // 2]
    var data_offset: Int     # Byte offset from start of data region.
    var data_size: Int       # Byte count in data region.

    fn __init__(out self):
        self.name = String("")
        self.dtype = String("")
        self.shape = List[Int]()
        self.data_offset = 0
        self.data_size = 0

    fn __init__(out self, name: String, dtype: String, shape: List[Int],
                data_offset: Int, data_size: Int):
        self.name = name
        self.dtype = dtype
        self.shape = shape.copy()
        self.data_offset = data_offset
        self.data_size = data_size

    fn __copyinit__(out self, copy: Self):
        self.name = copy.name
        self.dtype = copy.dtype
        self.shape = copy.shape.copy()
        self.data_offset = copy.data_offset
        self.data_size = copy.data_size

    fn __moveinit__(out self, deinit take: Self):
        self.name = take.name^
        self.dtype = take.dtype^
        self.shape = take.shape^
        self.data_offset = take.data_offset
        self.data_size = take.data_size

    fn num_elements(self) -> Int:
        if len(self.shape) == 0:
            return 1
        var n = 1
        for i in range(len(self.shape)):
            n *= self.shape[i]
        return n


struct SafetensorsFile(Copyable, Movable):
    """Opened safetensors file with parsed header."""
    var fd: Int32
    var file_size: Int
    var header_size: Int
    var data_start: Int
    var tensors: List[TensorInfo]

    fn __init__(out self):
        self.fd = -1
        self.file_size = 0
        self.header_size = 0
        self.data_start = 0
        self.tensors = List[TensorInfo]()

    fn __copyinit__(out self, copy: Self):
        self.fd = copy.fd
        self.file_size = copy.file_size
        self.header_size = copy.header_size
        self.data_start = copy.data_start
        self.tensors = copy.tensors.copy()

    fn __moveinit__(out self, deinit take: Self):
        self.fd = take.fd
        self.file_size = take.file_size
        self.header_size = take.header_size
        self.data_start = take.data_start
        self.tensors = take.tensors^

    fn close(var self):
        if self.fd >= 0:
            _fd_close(self.fd)

    fn find_tensor(self, name: String) -> Int:
        """Return index into self.tensors or -1."""
        for i in range(len(self.tensors)):
            if self.tensors[i].name == name:
                return i
        return -1

    fn has_tensor(self, name: String) -> Bool:
        return self.find_tensor(name) >= 0

    fn read_tensor_bytes(self, tensor_idx: Int) -> List[UInt8]:
        """Read raw bytes for tensor at index. Returns List[UInt8]."""
        var info = self.tensors[tensor_idx]
        var abs_offset = self.data_start + info.data_offset
        return _fd_read_into_list(self.fd, abs_offset, info.data_size)

    fn read_tensor_into(self, tensor_idx: Int, dst: UnsafePointer[UInt8]) -> Int:
        """Read raw bytes for tensor at index into pre-allocated dst."""
        var info = self.tensors[tensor_idx]
        var abs_offset = self.data_start + info.data_offset
        return _fd_pread(self.fd, abs_offset, dst, info.data_size)


fn _parse_safetensors_header(header_json: String) -> List[TensorInfo]:
    """Parse the safetensors JSON header into TensorInfo list.

    The JSON is a flat dict: tensor_name -> {dtype, shape, data_offsets}.
    """
    var result = List[TensorInfo]()
    var pos = 0
    var n = len(header_json)

    # Skip leading {
    while pos < n and _get_byte(header_json, pos) != ord("{"):
        pos += 1
    pos += 1

    while pos < n:
        # Find next quoted tensor name.
        while pos < n and _get_byte(header_json, pos) != ord('"'):
            if _get_byte(header_json, pos) == ord("}"):
                return result^
            pos += 1
        if pos >= n:
            break
        pos += 1  # skip opening quote
        var name_start = pos
        while pos < n and _get_byte(header_json, pos) != ord('"'):
            pos += 1
        var tensor_name = String(header_json[name_start:pos])
        pos += 1  # skip closing quote

        # Skip to value object { }.
        while pos < n and _get_byte(header_json, pos) != ord("{") and _get_byte(header_json, pos) != ord("}"):
            pos += 1
        if pos >= n or _get_byte(header_json, pos) == ord("}"):
            break

        # Extract the sub-object as a string.
        var obj_start = pos
        var depth = 0
        while pos < n:
            var b = _get_byte(header_json, pos)
            if b == ord("{"):
                depth += 1
            elif b == ord("}"):
                depth -= 1
                if depth == 0:
                    pos += 1
                    break
            pos += 1
        var obj_str = String(header_json[obj_start:pos])

        # Skip __metadata__ key.
        if tensor_name == "__metadata__":
            continue

        var dtype = _json_find_string(obj_str, "dtype")
        var shape = _json_find_int_array(obj_str, "shape")
        var offsets = _json_find_int_array(obj_str, "data_offsets")
        var data_offset = 0
        var data_size = 0
        if len(offsets) >= 2:
            data_offset = offsets[0]
            data_size = offsets[1] - offsets[0]

        result.append(TensorInfo(
            name=tensor_name,
            dtype=dtype,
            shape=shape^,
            data_offset=data_offset,
            data_size=data_size,
        ))

    return result^


fn open_safetensors(path: String) raises -> SafetensorsFile:
    """Open a safetensors file and parse its header.

    Format: 8-byte LE header_size | JSON header | raw tensor data.
    """
    var fd = _try_open_file(path)
    if fd < 0:
        raise Error("Failed to open: " + path)

    var file_size = _fd_file_size(fd)

    # Read 8-byte little-endian header size.
    var hdr_bytes = _fd_read_into_list(fd, 0, 8)
    if len(hdr_bytes) < 8:
        _fd_close(fd)
        raise Error("Failed to read safetensors header size")

    var header_size = 0
    for i in range(8):
        header_size |= Int(hdr_bytes[i]) << (i * 8)

    if header_size <= 0 or header_size > file_size - 8:
        _fd_close(fd)
        raise Error("Invalid safetensors header size: " + String(header_size))

    # Read JSON header.
    var json_bytes = _fd_read_into_list(fd, 8, header_size)
    if len(json_bytes) < header_size:
        _fd_close(fd)
        raise Error("Failed to read safetensors JSON header")

    # Build String from bytes.
    var header_json = String("")
    for i in range(header_size):
        header_json += chr(Int(json_bytes[i]))

    var tensors = _parse_safetensors_header(header_json)

    var result = SafetensorsFile()
    result.fd = fd
    result.file_size = file_size
    result.header_size = header_size
    result.data_start = 8 + header_size
    result.tensors = tensors^
    return result^


# ===----------------------------------------------------------------------=== #
# GGUF format (minimal reader for MXFP4_MOE)
# ===----------------------------------------------------------------------=== #

comptime GGUF_MAGIC: UInt32 = 0x46475547  # "GGUF" in LE
comptime GGUF_TYPE_UINT32: UInt32 = 4
comptime GGUF_TYPE_INT32: UInt32 = 5
comptime GGUF_TYPE_FLOAT32: UInt32 = 6
comptime GGUF_TYPE_STRING: UInt32 = 8
comptime GGUF_TYPE_ARRAY: UInt32 = 9
comptime GGUF_TYPE_UINT64: UInt32 = 10


struct GGUFTensorInfo(Copyable, Movable, ImplicitlyCopyable):
    """Metadata for one tensor inside a GGUF file."""
    var name: String
    var n_dims: Int
    var shape: List[Int]
    var tensor_type: UInt32
    var offset: Int

    fn __init__(out self):
        self.name = String("")
        self.n_dims = 0
        self.shape = List[Int]()
        self.tensor_type = 0
        self.offset = 0

    fn __copyinit__(out self, copy: Self):
        self.name = copy.name
        self.n_dims = copy.n_dims
        self.shape = copy.shape.copy()
        self.tensor_type = copy.tensor_type
        self.offset = copy.offset

    fn __moveinit__(out self, deinit take: Self):
        self.name = take.name^
        self.n_dims = take.n_dims
        self.shape = take.shape^
        self.tensor_type = take.tensor_type
        self.offset = take.offset


struct GGUFFile:
    """Opened GGUF file with parsed header."""
    var fd: Int32
    var file_size: Int
    var version: UInt32
    var n_tensors: Int
    var n_kv: Int
    var tensor_data_offset: Int
    var tensors: List[GGUFTensorInfo]
    var alignment: Int

    fn __init__(out self):
        self.fd = -1
        self.file_size = 0
        self.version = 0
        self.n_tensors = 0
        self.n_kv = 0
        self.tensor_data_offset = 0
        self.tensors = List[GGUFTensorInfo]()
        self.alignment = 32

    fn __moveinit__(out self, deinit take: Self):
        self.fd = take.fd
        self.file_size = take.file_size
        self.version = take.version
        self.n_tensors = take.n_tensors
        self.n_kv = take.n_kv
        self.tensor_data_offset = take.tensor_data_offset
        self.tensors = take.tensors^
        self.alignment = take.alignment

    fn close(var self):
        if self.fd >= 0:
            _fd_close(self.fd)

    fn find_tensor(self, name: String) -> Int:
        for i in range(len(self.tensors)):
            if self.tensors[i].name == name:
                return i
        return -1


fn _read_u32_at(fd: Int32, offset: Int) -> UInt32:
    """Read a little-endian uint32 at file offset."""
    var buf = _fd_read_into_list(fd, offset, 4)
    return UInt32(buf[0]) | (UInt32(buf[1]) << 8) | (UInt32(buf[2]) << 16) | (UInt32(buf[3]) << 24)


fn _read_u64_at(fd: Int32, offset: Int) -> UInt64:
    """Read a little-endian uint64 at file offset."""
    var buf = _fd_read_into_list(fd, offset, 8)
    var val = UInt64(0)
    for i in range(8):
        val |= UInt64(buf[i]) << (i * 8)
    return val


fn _read_gguf_string(fd: Int32, offset: Int) -> Tuple[String, Int]:
    """Read a GGUF string (u64 len + bytes). Returns (string, bytes_consumed)."""
    var slen = Int(_read_u64_at(fd, offset))
    var buf = _fd_read_into_list(fd, offset + 8, slen)
    var s = String("")
    for i in range(slen):
        s += chr(Int(buf[i]))
    return (s^, 8 + slen)


fn _skip_gguf_value(fd: Int32, offset: Int, vtype: UInt32) -> Int:
    """Skip a GGUF metadata value, returning bytes consumed."""
    if vtype == GGUF_TYPE_UINT32 or vtype == GGUF_TYPE_INT32 or vtype == GGUF_TYPE_FLOAT32:
        return 4
    elif vtype == GGUF_TYPE_UINT64:
        return 8
    elif vtype == GGUF_TYPE_STRING:
        var result = _read_gguf_string(fd, offset)
        return result[1]
    elif vtype == GGUF_TYPE_ARRAY:
        var elem_type = _read_u32_at(fd, offset)
        var count = Int(_read_u64_at(fd, offset + 4))
        var consumed = 12
        for _ in range(count):
            consumed += _skip_gguf_value(fd, offset + consumed, elem_type)
        return consumed
    return 0


fn open_gguf(path: String) raises -> GGUFFile:
    """Open a GGUF file and parse tensor metadata.

    GGUF v3 layout:
      magic(4) | version(4) | n_tensors(8) | n_kv(8)
      kv_pairs... | tensor_infos... | [padding] | tensor_data...
    """
    var fd = _try_open_file(path)
    if fd < 0:
        raise Error("Failed to open: " + path)

    var file_size = _fd_file_size(fd)

    var magic = _read_u32_at(fd, 0)
    if magic != GGUF_MAGIC:
        _fd_close(fd)
        raise Error("Not a GGUF file (bad magic)")

    var version = _read_u32_at(fd, 4)
    var n_tensors = Int(_read_u64_at(fd, 8))
    var n_kv = Int(_read_u64_at(fd, 16))
    var pos = 24

    # Skip KV metadata to reach tensor infos.
    var alignment = 32
    for _ in range(n_kv):
        var key_result = _read_gguf_string(fd, pos)
        var key_name = key_result[0]
        pos += key_result[1]
        var vtype = _read_u32_at(fd, pos)
        pos += 4
        if key_name == "general.alignment" and vtype == GGUF_TYPE_UINT32:
            alignment = Int(_read_u32_at(fd, pos))
        pos += _skip_gguf_value(fd, pos, vtype)

    # Parse tensor infos.
    var tensors = List[GGUFTensorInfo]()
    for _ in range(n_tensors):
        var info = GGUFTensorInfo()
        var name_result = _read_gguf_string(fd, pos)
        info.name = name_result[0]
        pos += name_result[1]
        info.n_dims = Int(_read_u32_at(fd, pos))
        pos += 4
        for _ in range(info.n_dims):
            info.shape.append(Int(_read_u64_at(fd, pos)))
            pos += 8
        info.tensor_type = _read_u32_at(fd, pos)
        pos += 4
        info.offset = Int(_read_u64_at(fd, pos))
        pos += 8
        tensors.append(info^)

    var tensor_data_offset = (pos + alignment - 1) // alignment * alignment

    var result = GGUFFile()
    result.fd = fd
    result.file_size = file_size
    result.version = version
    result.n_tensors = n_tensors
    result.n_kv = n_kv
    result.tensor_data_offset = tensor_data_offset
    result.tensors = tensors^
    result.alignment = alignment
    return result^


# ===----------------------------------------------------------------------=== #
# NVFP4 weight conversion and loading
# ===----------------------------------------------------------------------=== #


struct LoadedWeight(Copyable, Movable):
    """Packed NVFP4 weight data ready for the GEMM kernel.

    Access packed_weights and blockscales via .unsafe_ptr() for kernel calls.
    """
    var packed_weights: List[UInt8]   # [out, in//2] packed FP4 pairs.
    var blockscales: List[UInt8]      # [out, in//16] FP8 E4M3 bytes.
    var weight_scale_2: Float32
    var input_scale: Float32
    var out_features: Int
    var in_features: Int

    fn __init__(out self):
        self.packed_weights = List[UInt8]()
        self.blockscales = List[UInt8]()
        self.weight_scale_2 = 1.0
        self.input_scale = 1.0
        self.out_features = 0
        self.in_features = 0

    fn __init__(out self, out_features: Int, in_features: Int):
        var packed_size = out_features * (in_features // NVFP4_PACK_FACTOR)
        var scale_size = out_features * (in_features // NVFP4_GROUP_SIZE)
        self.packed_weights = List[UInt8](capacity=packed_size)
        for _ in range(packed_size):
            self.packed_weights.append(0)
        self.blockscales = List[UInt8](capacity=scale_size)
        for _ in range(scale_size):
            self.blockscales.append(0)
        self.weight_scale_2 = 1.0
        self.input_scale = 1.0
        self.out_features = out_features
        self.in_features = in_features

    fn __copyinit__(out self, copy: Self):
        self.packed_weights = copy.packed_weights.copy()
        self.blockscales = copy.blockscales.copy()
        self.weight_scale_2 = copy.weight_scale_2
        self.input_scale = copy.input_scale
        self.out_features = copy.out_features
        self.in_features = copy.in_features

    fn __moveinit__(out self, deinit take: Self):
        self.packed_weights = take.packed_weights^
        self.blockscales = take.blockscales^
        self.weight_scale_2 = take.weight_scale_2
        self.input_scale = take.input_scale
        self.out_features = take.out_features
        self.in_features = take.in_features

    fn to_linear(self) -> NvFp4Linear:
        """Create NvFp4Linear metadata (pass .unsafe_ptr() separately)."""
        var lin = NvFp4Linear(self.in_features, self.out_features)
        lin.input_scale = self.input_scale
        lin.weight_scale_2 = self.weight_scale_2
        return lin^

    fn packed_weight_bytes(self) -> Int:
        return self.out_features * (self.in_features // NVFP4_PACK_FACTOR)

    fn blockscale_bytes(self) -> Int:
        return self.out_features * (self.in_features // NVFP4_GROUP_SIZE)


fn _read_f32_from_bytes(buf: List[UInt8]) -> Float32:
    """Interpret first 4 bytes of buf as little-endian IEEE754 float32."""
    if len(buf) < 4:
        return 1.0
    var bits = UInt32(buf[0]) | (UInt32(buf[1]) << 8) | (UInt32(buf[2]) << 16) | (UInt32(buf[3]) << 24)
    var sign = Int(bits >> 31)
    var exp_bits = Int((bits >> 23) & 0xFF)
    var mantissa = Int(bits & 0x7FFFFF)
    if exp_bits == 0:
        return Float32(-0.0) if sign == 1 else Float32(0.0)
    if exp_bits == 255:
        return Float32(1.0)  # NaN/Inf fallback
    var frac = Float32(1.0) + Float32(mantissa) / Float32(1 << 23)
    var result = frac
    var exp_val = exp_bits - 127
    # Apply 2^exp via repeated multiply (no pow() available).
    if exp_val > 0:
        for _ in range(exp_val):
            result *= 2.0
    elif exp_val < 0:
        for _ in range(-exp_val):
            result *= 0.5
    return -result if sign == 1 else result


fn load_linear_safetensors(
    sf: SafetensorsFile,
    prefix: String,
) raises -> LoadedWeight:
    """Load one NVFP4 linear layer from safetensors.

    Expects these tensors:
      {prefix}.weight          — [out, in//2] dtype U8 (packed FP4 pairs)
      {prefix}.weight_scale    — [out, in//16] dtype F8_E4M3 (blockscales)
      {prefix}.weight_scale_2  — [] dtype F32 (second-level calibration scale)
      {prefix}.input_scale     — [] dtype F32 (activation scale)
    """
    var w_name = prefix + ".weight"
    var s_name = prefix + ".weight_scale"
    var s2_name = prefix + ".weight_scale_2"
    var is_name = prefix + ".input_scale"

    var w_idx = sf.find_tensor(w_name)
    if w_idx < 0:
        raise Error("Tensor not found: " + w_name)
    var s_idx = sf.find_tensor(s_name)
    if s_idx < 0:
        raise Error("Tensor not found: " + s_name)

    var w_info = sf.tensors[w_idx]
    if len(w_info.shape) < 2:
        raise Error("Weight tensor must be 2D: " + w_name)

    var out_features = w_info.shape[0]
    var packed_k = w_info.shape[1]
    var in_features = packed_k * NVFP4_PACK_FACTOR

    var result = LoadedWeight(out_features, in_features)

    # Read packed weights directly into pre-allocated list.
    var n = sf.read_tensor_into(w_idx, result.packed_weights.unsafe_ptr())
    if n != result.packed_weight_bytes():
        raise Error("Weight read size mismatch for " + w_name
                     + ": expected " + String(result.packed_weight_bytes())
                     + ", got " + String(n))

    # Read blockscales.
    n = sf.read_tensor_into(s_idx, result.blockscales.unsafe_ptr())
    if n != result.blockscale_bytes():
        raise Error("Scale read size mismatch for " + s_name)

    # Read scalar scales (optional — default 1.0).
    var s2_idx = sf.find_tensor(s2_name)
    if s2_idx >= 0:
        var s2_bytes = sf.read_tensor_bytes(s2_idx)
        result.weight_scale_2 = _read_f32_from_bytes(s2_bytes)

    var is_idx = sf.find_tensor(is_name)
    if is_idx >= 0:
        var is_bytes = sf.read_tensor_bytes(is_idx)
        result.input_scale = _read_f32_from_bytes(is_bytes)

    return result^


# ===----------------------------------------------------------------------=== #
# Qwen 3.5 model structure
# ===----------------------------------------------------------------------=== #

# Qwen 3.5 naming conventions in safetensors:
#   model.layers.{l}.self_attn.{q,k,v,o}_proj.{weight,weight_scale,...}
#   model.layers.{l}.mlp.gate.weight              (router, BF16)
#   model.layers.{l}.mlp.experts.{e}.{gate,up,down}_proj.{...}
#   model.layers.{l}.mlp.shared_expert.{gate,up,down}_proj.{...}
#   model.layers.{l}.{input,post_attention}_layernorm.weight (BF16)
#   model.embed_tokens.weight  /  model.norm.weight  /  lm_head.weight


struct Qwen35LayerWeights(Movable):
    """Weights for one transformer layer in Qwen 3.5."""
    # Attention projections (NVFP4 quantized).
    var q_proj: LoadedWeight
    var k_proj: LoadedWeight
    var v_proj: LoadedWeight
    var o_proj: LoadedWeight
    # MoE expert projections (NVFP4 quantized): [num_experts] each.
    var expert_gate: List[LoadedWeight]
    var expert_up: List[LoadedWeight]
    var expert_down: List[LoadedWeight]
    # Shared expert (NVFP4 quantized).
    var shared_gate: LoadedWeight
    var shared_up: LoadedWeight
    var shared_down: LoadedWeight
    # Router gate weights (BF16, raw bytes).
    var router_weight: List[UInt8]
    # Layer norms (BF16, raw bytes).
    var input_layernorm: List[UInt8]
    var post_attn_layernorm: List[UInt8]

    fn __init__(out self):
        self.q_proj = LoadedWeight()
        self.k_proj = LoadedWeight()
        self.v_proj = LoadedWeight()
        self.o_proj = LoadedWeight()
        self.expert_gate = List[LoadedWeight]()
        self.expert_up = List[LoadedWeight]()
        self.expert_down = List[LoadedWeight]()
        self.shared_gate = LoadedWeight()
        self.shared_up = LoadedWeight()
        self.shared_down = LoadedWeight()
        self.router_weight = List[UInt8]()
        self.input_layernorm = List[UInt8]()
        self.post_attn_layernorm = List[UInt8]()

    fn __moveinit__(out self, deinit take: Self):
        self.q_proj = take.q_proj^
        self.k_proj = take.k_proj^
        self.v_proj = take.v_proj^
        self.o_proj = take.o_proj^
        self.expert_gate = take.expert_gate^
        self.expert_up = take.expert_up^
        self.expert_down = take.expert_down^
        self.shared_gate = take.shared_gate^
        self.shared_up = take.shared_up^
        self.shared_down = take.shared_down^
        self.router_weight = take.router_weight^
        self.input_layernorm = take.input_layernorm^
        self.post_attn_layernorm = take.post_attn_layernorm^


fn _find_load_linear(
    shards: List[SafetensorsFile], prefix: String
) raises -> LoadedWeight:
    """Find which shard has a tensor and load the NVFP4 linear from it."""
    var w_name = prefix + ".weight"
    for i in range(len(shards)):
        if shards[i].has_tensor(w_name):
            return load_linear_safetensors(shards[i], prefix)
    raise Error("No shard contains " + w_name)


fn _find_load_raw(
    shards: List[SafetensorsFile], name: String
) raises -> List[UInt8]:
    """Find which shard has a tensor and load its raw bytes."""
    for i in range(len(shards)):
        var idx = shards[i].find_tensor(name)
        if idx >= 0:
            return shards[i].read_tensor_bytes(idx)
    raise Error("No shard contains " + name)


fn load_qwen35_layer(
    shards: List[SafetensorsFile],
    layer_idx: Int,
    num_experts: Int,
) raises -> Qwen35LayerWeights:
    """Load all weights for one Qwen 3.5 transformer layer.

    Tensors may be spread across multiple safetensor shards.

    Args:
        shards: List of opened safetensors shard files.
        layer_idx: Layer index (0-based).
        num_experts: Number of routed experts (e.g. 512 for Qwen 3.5 397B).
    """
    var layer = Qwen35LayerWeights()
    var lp = "model.layers." + String(layer_idx)

    # Attention projections.
    layer.q_proj = _find_load_linear(shards, lp + ".self_attn.q_proj")
    layer.k_proj = _find_load_linear(shards, lp + ".self_attn.k_proj")
    layer.v_proj = _find_load_linear(shards, lp + ".self_attn.v_proj")
    layer.o_proj = _find_load_linear(shards, lp + ".self_attn.o_proj")

    # MoE experts.
    for e in range(num_experts):
        var ep = lp + ".mlp.experts." + String(e)
        layer.expert_gate.append(_find_load_linear(shards, ep + ".gate_proj"))
        layer.expert_up.append(_find_load_linear(shards, ep + ".up_proj"))
        layer.expert_down.append(_find_load_linear(shards, ep + ".down_proj"))

    # Shared expert.
    var sp = lp + ".mlp.shared_expert"
    layer.shared_gate = _find_load_linear(shards, sp + ".gate_proj")
    layer.shared_up = _find_load_linear(shards, sp + ".up_proj")
    layer.shared_down = _find_load_linear(shards, sp + ".down_proj")

    # Router gate (BF16, not quantized).
    layer.router_weight = _find_load_raw(shards, lp + ".mlp.gate.weight")

    # Layer norms.
    layer.input_layernorm = _find_load_raw(shards, lp + ".input_layernorm.weight")
    layer.post_attn_layernorm = _find_load_raw(shards, lp + ".post_attention_layernorm.weight")

    return layer^


# ===----------------------------------------------------------------------=== #
# Multi-shard discovery
# ===----------------------------------------------------------------------=== #


fn _zero_pad5(n: Int) -> String:
    """Zero-pad an integer to 5 digits (00001-99999)."""
    var s = String(n)
    while len(s) < 5:
        s = "0" + s
    return s^


fn discover_safetensor_shards(model_dir: String) raises -> List[String]:
    """Find safetensor shard files in a model directory.

    Looks for model.safetensors (single) or model-NNNNN-of-NNNNN.safetensors.
    """
    var paths = List[String]()

    # Try single file first.
    var single = model_dir + "/model.safetensors"
    var fd = _try_open_file(single)
    if fd >= 0:
        _fd_close(fd)
        paths.append(single)
        return paths^

    # Probe sharded files by trying each shard index.
    for i in range(1, 200):
        var found = False
        for total in range(i, 200):
            var path = model_dir + "/model-" + _zero_pad5(i) + "-of-" + _zero_pad5(total) + ".safetensors"
            var sfd = _try_open_file(path)
            if sfd >= 0:
                _fd_close(sfd)
                paths.append(path)
                found = True
                break
            if len(paths) > 0:
                break  # Already know the total from previous shard.
        if not found and len(paths) > 0:
            break
        if not found and i > 1:
            break

    if len(paths) == 0:
        raise Error("No safetensors files found in " + model_dir)

    return paths^


fn open_model_shards(model_dir: String) raises -> List[SafetensorsFile]:
    """Discover and open all safetensor shards for a model directory."""
    var paths = discover_safetensor_shards(model_dir)
    var files = List[SafetensorsFile]()
    for i in range(len(paths)):
        files.append(open_safetensors(paths[i]))
    return files^


# ===----------------------------------------------------------------------=== #
# Standalone verification — run as: mojo build weight_loader.mojo && ./weight_loader <file>
# ===----------------------------------------------------------------------=== #


fn main() raises:
    """Parse a safetensors or GGUF file and print tensor metadata."""
    from sys import argv

    if len(argv()) < 2:
        print("Usage: weight_loader <path.safetensors|path.gguf>")
        return

    var path = argv()[1]
    print("File:", path)

    if path.endswith(".gguf"):
        var gguf = open_gguf(path)
        print("  Version:", gguf.version, " Tensors:", gguf.n_tensors,
              " KV:", gguf.n_kv, " Alignment:", gguf.alignment)
        var limit = 20
        if gguf.n_tensors < limit:
            limit = gguf.n_tensors
        for i in range(limit):
            print("   ", gguf.tensors[i].name, "type=", gguf.tensors[i].tensor_type)
        gguf^.close()
    elif path.endswith(".safetensors"):
        var sf = open_safetensors(path)
        print("  Header:", sf.header_size, "bytes  Tensors:", len(sf.tensors))
        for i in range(len(sf.tensors)):
            print("   ", sf.tensors[i].name, "dtype=", sf.tensors[i].dtype,
                  "size=", sf.tensors[i].data_size)
        # Load first NVFP4 linear found.
        for i in range(len(sf.tensors)):
            if sf.tensors[i].name.endswith(".weight") and sf.tensors[i].dtype == "U8":
                var prefix = String(sf.tensors[i].name[:len(sf.tensors[i].name) - 7])
                if sf.has_tensor(prefix + ".weight_scale"):
                    var w = load_linear_safetensors(sf, prefix)
                    print("  Loaded:", prefix, "out=", w.out_features,
                          "in=", w.in_features, "ws2=", w.weight_scale_2,
                          "is=", w.input_scale)
                    break
        sf^.close()

    print("Done.")
