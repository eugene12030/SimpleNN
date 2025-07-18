# SimpleNN

A simple neural network implementation in Haskell using `cabal`, featuring an interactive command-line interface.

---

## 📦 Installation

To get started with SimpleNN, follow these steps:

1. **Prerequisites**: Ensure you have **GHC (Glasgow Haskell Compiler)** and **Cabal (Haskell Build Tool)** installed. The easiest way to get them is via `ghcup`:

    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | sh
    ```

    Follow the on-screen instructions.

2. **Clone the repository**:

    ```bash
    git clone your-repository-url # Replace with your actual repository URL
    cd SimpleNN # Navigate into the project directory
    ```

3. **Fetch dependencies and build**:

    ```bash
    cabal update
    cabal build
    ```

    This command will download all required packages and compile your project, creating the `snn` executable.

---

## 🚀 Usage

SimpleNN operates primarily through an interactive shell. You initiate the shell, and then run commands within it.

### Starting the Interactive Shell

To begin, run the shell from your terminal:

```bash
cabal run SimpleNN -- shell
```

You will see the prompt `>>`.

---

### 🧠 Commands within the Shell

#### `init` - Initialize Network

Initializes a new neural network architecture. You must run this command first to create a network instance.

**Signature:**

```shell
init -i INPUT_SIZE -l LAYER_SIZE... -a ACTIVATION...
```

- `-i INPUT_SIZE`: The number of input features (e.g., 784 for MNIST images).
- `-l LAYER_SIZE...`: Specify the size for each hidden and output layer. Repeat this flag for each layer.
- `-a ACTIVATION...`: Specify the activation function for each corresponding layer. Repeat this flag to match the number of `-l` flags. Supported activations: `relu`, `softmax`, `identity`.

**Example:**

```shell
init -i 784 -l 128 -l 10 -a relu -a softmax
```

---

#### `train` - Train Network

Trains the currently initialized network using a CSV dataset.

**Signature:**

```shell
train --train CSV_FILE [--valid-ratio R] [-e EPOCHS] [--lr RATE]
```

- `--train CSV_FILE`: Path to the CSV file containing your training data. The first column is expected to be the label, followed by features.
- `--valid-ratio R` *(optional)*: The fraction of the dataset to reserve for validation (default: `0.8`).
- `-e EPOCHS` *(optional)*: The number of training epochs.
- `--lr RATE` *(optional)*: The learning rate for optimization.

**Example:**

```shell
train --train mnist_train.csv -e 10 --lr 0.01
```

---

#### `predict` - Make Predictions

Uses the currently trained network to make predictions on new input data from a CSV file. The predictions are printed to the console and saved to `prediction_result.csv`.

**Signature:**

```shell
predict --input CSV_FILE
```

- `--input CSV_FILE`: Path to the CSV file with new data for prediction. The first column (if present) will be ignored; subsequent columns are treated as features.

**Example:**

```shell
predict --input new_images_to_predict.csv
```

---

#### `eval` - Evaluate Network

Evaluates the accuracy of the currently trained network against a test CSV dataset.

**Signature:**

```shell
eval --test CSV_FILE
```

- `--test CSV_FILE`: Path to the CSV file with test data. The first column is expected to be the true label, followed by features.

**Example:**

```shell
eval --test mnist_test.csv
```

---

#### `show` - Display Network

Displays the structure and current parameters of the neural network initialized in the shell session.

**Signature:**

```shell
show
```

---

#### `help` - Show Help

Displays a summary of available commands and their basic usage within the shell.

**Signature:**

```shell
help
```

---

#### `exit` / `quit` - Exit Shell

Exits the interactive SimpleNN shell.

**Signature:**

```shell
exit
# or
quit
```

---

## 📌 Example Workflow in Shell

Assuming you have `mnist_train.csv` and `mnist_test.csv` in your project’s root directory:

1. **Start the shell:**

    ```bash
    cabal run SimpleNN -- shell
    ```

    You will see:  
    `=== SimpleNN Interactive Shell ===`  
    followed by `>>`.

2. **Initialize the model:**

    ```shell
    init -i 784 -l 128 -l 10 -a relu -a softmax
    ```

    This sets up a network with 784 inputs, one hidden layer of 128 neurons (ReLU), and an output layer of 10 neurons (Softmax).

3. **Train the model:**

    ```shell
    train --train mnist_train.csv -e 10 --lr 0.01
    ```

    The network will train for 10 epochs with a learning rate of 0.01. This might take time depending on your dataset and hardware. You'll see `Training complete` when it's done.

4. **Evaluate the model's accuracy:**

    ```shell
    eval --test mnist_test.csv
    ```

    The shell will output the number of test samples, correct predictions, and overall accuracy.

5. **Make predictions on new data:**

    ```shell
    predict --input mnist_test.csv
    ```

    The predicted labels will be printed to your console and saved to `prediction_result.csv`.

6. **Exit the shell:**

    ```shell
    exit
    ```

---

## 🗂️ Project Structure

- `Core.hs`: Contains the fundamental building blocks of the neural network, including vector/matrix operations, layer definitions, and the core forward/backward propagation algorithms.
- `CLI.hs`: Defines the command-line interface, parses user input, and manages the interactive shell's state (including the neural network instance).
- `DataProcessing.hs`: Handles reading and parsing CSV data, normalizing pixel values, splitting datasets into training and validation sets, and writing prediction results to CSV.

---

Feel free to customize the network architecture, training parameters, and experiment with different datasets!  
If you have any questions or run into issues, don’t hesitate to ask.
