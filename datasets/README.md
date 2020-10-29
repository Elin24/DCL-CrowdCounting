Dataset processing

## Support
- SHHA
- SHHB
- UCF-QNRF
- UCF-CC-50
- GCC

## Generation:

1. Download processed dataset from relevant web site.
2. modify the line 53 in `generate_data.py` to specify the dataset.
3. modify `root` and `nroot` to specify the original dataset path and processed dataset path.
4. modify setting in `SHHA` to specifiy where the dataloader reads data.