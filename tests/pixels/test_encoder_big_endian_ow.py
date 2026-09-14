"""Synthetic regressions for Dataset byte order at the encoder boundary."""

from copy import deepcopy
from io import BytesIO

import pytest

from pydicom import dcmread, dcmwrite
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.pixels.encoders import RLELosslessEncoder
from pydicom.pixels.encoders.base import EncodeRunner
from pydicom.uid import (
    ExplicitVRBigEndian,
    ExplicitVRLittleEndian,
    RLELossless,
    SecondaryCaptureImageStorage,
    generate_uid,
)


def pixel_fixture(vr, frames, rows, columns, big_endian=True, bits=8):
    """Author samples before independently serializing their storage words."""
    values = list(range(1, frames * rows * columns + 1))
    canonical = b"".join(x.to_bytes(bits // 8, "little") for x in values)
    canonical += b"\x00" if len(canonical) % 2 else b""
    stored = canonical
    if big_endian and (vr == "OW" or bits > 8):
        stored = b"".join(
            canonical[i : i + 2][::-1] for i in range(0, len(canonical), 2)
        )

    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = (
        ExplicitVRBigEndian if big_endian else ExplicitVRLittleEndian
    )
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = generate_uid()
    ds.Rows, ds.Columns, ds.NumberOfFrames = rows, columns, frames
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = bits
    ds.HighBit, ds.PixelRepresentation = bits - 1, 0
    ds.add_new(0x7FE00010, vr, stored)
    return ds, values, canonical


@pytest.mark.parametrize("vr", ["OB", "OW"])
@pytest.mark.parametrize("big_endian", [False, True])
@pytest.mark.parametrize("frames,columns", [(1, 4), (1, 3), (2, 3), (3, 3)])
def test_runner_word_order(vr, big_endian, frames, columns):
    ds, _, canonical = pixel_fixture(vr, frames, 1, columns, big_endian)
    original = deepcopy(ds)
    runner = EncodeRunner(RLELossless)
    runner.set_source(ds)
    assert runner.src == canonical
    if not big_endian or vr == "OB":
        assert runner.src is ds.PixelData
    for index in range(frames):
        assert (
            runner.get_frame(index)
            == canonical[index * columns : (index + 1) * columns]
        )
    assert ds == original


@pytest.mark.parametrize("vr,bits", [("OB", 8), ("OW", 8), ("OW", 16)])
@pytest.mark.parametrize("plugin", ["pydicom", "pylibjpeg"])
@pytest.mark.parametrize("big_endian", [False, True])
@pytest.mark.parametrize("frames,side", [(1, 2), (1, 3), (2, 3), (3, 3)])
def test_rle_file_roundtrip(vr, bits, plugin, big_endian, frames, side):
    np = pytest.importorskip("numpy")
    if plugin not in RLELosslessEncoder.available_plugins:
        pytest.skip(f"Optional RLE encoder {plugin} is unavailable")
    ds, values, _ = pixel_fixture(vr, frames, side, side, big_endian, bits)
    shape = (frames, side, side) if frames > 1 else (side, side)
    expected = np.array(values, dtype=f"uint{bits}").reshape(shape)
    # Establish that the authored bytes are valid before invoking the encoder.
    np.testing.assert_array_equal(ds.pixel_array, expected)
    ds.compress(RLELossless, encoding_plugin=plugin, generate_instance_uid=False)
    assert ds.file_meta.TransferSyntaxUID == RLELossless
    np.testing.assert_array_equal(ds.pixel_array, expected)
    stream = BytesIO()
    dcmwrite(stream, ds, enforce_file_format=True)
    stream.seek(0)
    restored = dcmread(stream)
    assert restored.file_meta.TransferSyntaxUID == RLELossless
    np.testing.assert_array_equal(restored.pixel_array, expected)


@pytest.mark.parametrize("kind", [bytes, bytearray, memoryview])
def test_raw_buffer_has_no_dataset_word_order(kind):
    source = kind(b"\x02\x01\x04\x03\x06\x05")
    runner = EncodeRunner(RLELossless)
    runner.set_source(source)
    assert runner.src is source


def test_short_buffer_retains_validation_error():
    ds, _, _ = pixel_fixture("OW", 2, 3, 3)
    ds.PixelData = ds.PixelData[:4]
    runner = EncodeRunner(RLELossless)
    runner.set_source(ds)
    with pytest.raises(ValueError, match="length of the uncompressed pixel data"):
        runner.validate()
