# Setup
First you would want to create a data
dir I prefer naming it `data` but 
you can name as you like. 

then you would want to move `part1`,
`part2`,and `metadata.csv` inside the 
data dir you just made.

Then install the project pip dependencies

```console
pip3 install -r requirements.txt
```

After installing torch ensure that cuda
device is available, for speed. If not
manually install torch from the [torch 
website](https://pytorch.org/get-started/locally/)

# Running the main script

Inside the main script, in the `__name__ == '__main__'` you can add the functions
which you want to run (e.g. `train_simple()`,`train_resnet()` , etc.)
and then run the script:
```console
python3 main.py
```

All the functions that execute the assignment's
code are in comments, uncomment the ones you want to run.