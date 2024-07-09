# SSGReps

This Python program can be used to compute representations of Spin-Space Groups (SSG).

## Getting Started

1. **Clone the repository:**

    ```bash
    git clone https://github.com/zine-phy/SSGReps.git
    ```

2. **Extract the data in `ssg_data`:**

    ```bash
    tar -xzvf identify.pkl.tar.gz
    ```

3. **Install the necessary libraries:**

    Generally, installing `pymatgen` will install all the dependencies.

    ```bash
    pip install pymatgen==2022.0.17
    ```

## Usage

Run the main program `SSGReps.py`:


    python SSGReps.py --ssgNum 194.1.6.1.P --kp 0 0 0.5 --out rep_matrix --groupType 1 > ssgrep.out
    python SSGReps.py --ssgNum 194.1.6.1.P --kp 0 0 0.5 > ssgrep.out

### Parameters

- `--ssgNum` (required): Specifies the SSG number.
- `--kp` (required): Specifies the k-point (the k-point is defined as the primitive cell with pure translation operations of the SSG).
- `--out` (optional): Specifies the output type, which can be `charactor` (default), `rep_degree`, or `rep_matrix`. 
  - `charactor`: Outputs the character table.
  - `rep_degree`: Outputs the degree of the representation.
  - `rep_matrix`: Outputs the representation matrix.
- `--groupType` (optional): Specifies the group type, either `2` for double group (default) or `1` for single group.
