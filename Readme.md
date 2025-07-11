# SimpleNN

A simple neural network implementation in Haskell using `cabal`.

## 📦 Installation

To install the project, navigate to the project directory and run in the terminal:

```bash
cabal install
```

To create the executable file, run:

```bash
cabal update
cabal build
```

After building, you can use `cabal run` to execute the program.

## 🚀 Usage

### Initialization

```bash
cabal run SimpleNN -- init -i INPUT_SIZE -l LAYER_SIZE [-l LAYER_SIZE ...] -a ACTIVATION [-a ACTIVATION ...]
```

### Training

```bash
cabal run SimpleNN -- train -i INPUT_SIZE -l LAYER_SIZE [-l LAYER_SIZE ...] -a ACTIVATION [-a ACTIVATION ...] --train CSV_FILE [--valid-ratio R] -e EPOCHS --lr RATE
```

### Prediction

```bash
cabal run SimpleNN -- predict -i INPUT_SIZE -l LAYER_SIZE [-l LAYER_SIZE ...] -a ACTIVATION [-a ACTIVATION ...] --input CSV_FILE
```

### Evaluation

```bash
cabal run SimpleNN -- eval -i INPUT_SIZE -l LAYER_SIZE [-l LAYER_SIZE ...] -a ACTIVATION [-a ACTIVATION ...] --test CSV_FILE
```

## 📌 Example Script

Assuming the file `mnist_train.csv` is in the root directory:

1. Initialize the model:
   ```bash
   cabal run SimpleNN -- init -i 784 -l 16 -l 10 -a ReLU -a Softmax
   ```

2. Train the model:
   ```bash
   cabal run SimpleNN -- train -i 784 -l 16 -l 10 -a ReLU -a Softmax --train mnist_train.csv -e 10 --lr 0.01
   ```

3. Make predictions:
   ```bash
   cabal run SimpleNN -- predict -i 784 -l 16 -l 10 -a ReLU -a Softmax --input mnist_train.csv
   ```

4. Evaluate the model:
   ```bash
   cabal run SimpleNN -- eval -i 784 -l 16 -l 10 -a ReLU -a Softmax --test mnist_train.csv
   ```

---

Feel free to customize the layer sizes and activation functions as needed.
