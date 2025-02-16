## 🚀 Setup

To ensure compatibility, this project uses **llama.cpp version b4514**``.

### 1️⃣ Clone the Repository

```sh
cd /path/to/github/directory
```

### 2️⃣ Setup `llama.cpp`

```sh
cd external
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout b4514
git switch -c b4514
git branch  # Ensure the branch is `b4514` or lower
git submodule update --init --recursive
git lfs install
git lfs pull
```

## 🛠️ Build Instructions

### 1️⃣ Create Build Directory

```sh
cd ../..
mkdir build && cd build
```

### 2️⃣ Compile the Project

```sh
cmake ..
make
```

## ✅ Running Tests

After a successful build, run the test binary:

```sh
./bin/test
```

