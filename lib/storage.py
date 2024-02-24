from pathlib import Path
import numpy as np
import h5py as h5

default_h5_opts = dict(compression="gzip", compression_opts=9)

def quantize(data, least_significant_digit):
    """
    copied from netCDF4/utils.py (v1.6.3)

    quantize data to improve compression. data is quantized using
    around(scale*data)/scale, where scale is 2**bits, and bits is determined
    from the least_significant_digit. For example, if
    least_significant_digit=1, bits will be 4.
    """
    precision = pow(10.,-least_significant_digit)
    exp = np.log10(precision)
    if exp < 0:
        exp = int(np.floor(exp))
    else:
        exp = int(np.ceil(exp))
    bits = np.ceil(np.log2(pow(10.,-exp)))
    scale = pow(2.,bits)
    datout = np.around(scale*data)/scale
    return datout


def get_storage_functions(save_file_path, overwrite=False):
    save_file_path = Path(save_file_path)
    save_file_path.parent.mkdir(exist_ok=True)

    def check_equals_saved(val, val_saved, name):
        try:
            try:
                assert np.array_equal(val, val_saved, equal_nan=True)
            except TypeError as e:
                if "ufunc 'isnan' not supported" not in str(e):
                    raise e
                assert np.array_equal(val, val_saved)
        except AssertionError:
            assert np.array_equal(val_saved, np.array(val, dtype=bytes)),\
                f'inconsistent value in dataset/attr ' +\
                f'{name} in file {save_file_path}'

    def save_results(dset_name, data, overwrite=overwrite, attrs=None, **kwargs):
        with h5.File(save_file_path, 'a') as f:
            try:
                if isinstance(data, np.ndarray) and data.size > 1:
                    # only apply default compression settings
                    # to non-singleton arrays
                    kwargs.update(default_h5_opts)
                dset = f.create_dataset(dset_name, data=data, **kwargs)
                if attrs is not None:
                    assert hasattr(attrs, '__getitem__'), type(attrs)
                    dset.attrs.update(attrs)
            except (ValueError, RuntimeError, KeyError) as err:
                # if the dataset to save already exists, check that the
                #   saved value equals the value to save;
                #   if the error is something else, raise it
                if ('name already exists' not in str(err)
                        and "Can't open attribute" not in str(err)):
                    raise err

                dset = f[dset_name]
                try:
                    # always check equivalence, even when overwriting
                    # (deleting an h5 dataset does not free up space,
                    # so unneeded updates waste it)
                    check_equals_saved(data, dset[()], dset_name)
                    if attrs is not None:
                        assert hasattr(attrs, '__getitem__'), type(attrs)
                        for k, v in attrs.items():
                            check_equals_saved(v, dset.attrs[k], f'{dset_name}.{k}')
                except (AssertionError, KeyError) as err:
                    # dataset exists and does not match data to be written
                    # some attrs may be missing
                    if overwrite:
                        del f[dset_name]
                        save_results(dset_name, data, attrs=attrs, **kwargs)
                    else:
                        raise err

    def add_attr_to_dset(dset_name, attrs):
        with h5.File(save_file_path, 'a') as f:
            try:
                dset = f[dset_name]
            except KeyError:
                raise KeyError(
                    f'dataset or group {dset_name} does not exist; '
                    'cannot add attr')
            for k, v in attrs.items():
                if k in dset.attrs:
                    check_equals_saved(v, dset.attrs[k], f'{dset_name}.{k}')
                else:
                    dset.attrs[k] = v

    def link_dsets(src_name, dst_name, overwrite=overwrite):
        if dst_name.find('/') != 0:
            dst_name = '/' + dst_name
        with h5.File(save_file_path, 'a') as f:
            assert dst_name in f, \
                f'cannot create link to {dst_name}, which does not exist'
            if src_name in f:
                l2 = f.get(src_name, getlink=True)
                if isinstance(l2, h5.SoftLink) and l2.path == dst_name:
                    return
                if not overwrite:
                    raise RuntimeError(
                        f'cannot create link at {src_name}; '
                        f'dataset already exists (node type {type(l2)})')
                del f[src_name]

            f[src_name] = h5.SoftLink(dst_name)

    def copy_group(src_file_or_group, src, dst_name=None):
        """
        From the h5py docs:
        src – What to copy. May be a path in the file or a Group/Dataset object.
        dst_name – If the destination is a Group object, use this for the name of the copied object (default is basename).
        """
        with h5.File(save_file_path, 'a') as f:
            if isinstance(src, str):
                src_obj = src_file_or_group[src]
            else:
                src_obj = src

            if dst_name is None:
                dst_name = src_obj.name

            if dst_name not in f:
                src_file_or_group.copy(src, f, name=dst_name)
                return

            # check for existing
            dst = f[dst_name]
            assert isinstance(dst, type(src_obj))  # Dataset or Group

        # if dst exists
        if isinstance(src_obj, h5.Group):
            # copy/check recursively
            for k, obj in src_obj.items():
                copy_group(src_obj, k, dst_name=None)
        else:
            # write data with equality check
            save_results(dst_name, src_obj[()], attrs=src_obj.attrs)

    return save_results, add_attr_to_dset, check_equals_saved, link_dsets, copy_group
