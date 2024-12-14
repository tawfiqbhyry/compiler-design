# TEAM Language Compiler and Interpreter

Welcome to the **TEAM Language Compiler and Interpreter**! This project provides a comprehensive toolchain for the TEAM programming language, including lexical analysis, parsing, semantic analysis, and execution. Additionally, it features a user-friendly graphical interface for code input, visualization, and execution.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Language Syntax](#language-syntax)
  - [Data Types](#data-types)
  - [Variables and Constants](#variables-and-constants)
  - [Arrays](#arrays)
  - [Functions](#functions)
  - [Control Structures](#control-structures)
  - [Classes](#classes)
  - [Exception Handling](#exception-handling)
  - [Built-in Functions](#built-in-functions)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the GUI](#running-the-gui)
  - [Using the CLI](#using-the-cli)
- [Examples](#examples)
- [Architecture](#architecture)
  - [Tokenizer](#tokenizer)
  - [Parser](#parser)
  - [Abstract Syntax Tree (AST)](#abstract-syntax-tree-ast)
  - [Symbol Table](#symbol-table)
  - [Backend Execution](#backend-execution)
  - [Graphical User Interface](#graphical-user-interface)
- [Error Handling](#error-handling)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

The TEAM Language Compiler and Interpreter is a complete toolchain for the custom TEAM programming language. It allows users to write, analyze, visualize, and execute TEAM code seamlessly through a graphical user interface (GUI) built with Tkinter. The project includes:

- **Lexical Analysis**: Tokenizes the source code based on defined token specifications.
- **Parsing**: Analyzes the token sequence against the grammar to build an Abstract Syntax Tree (AST).
- **Semantic Analysis**: Constructs symbol tables to manage scope and detect semantic errors.
- **Execution**: Interprets the AST to execute the TEAM code.
- **GUI**: Provides an intuitive interface for code input, visualization, and output display.

## Features

- **Comprehensive Tokenizer**: Recognizes comments, keywords, data types, literals, identifiers, operators, and delimiters.
- **Robust Parser**: Implements a recursive descent parser based on the defined grammar.
- **Semantic Analysis**: Manages symbol tables for scope tracking and error detection.
- **Execution Engine**: Interprets and executes TEAM code, supporting variables, arrays, functions, classes, and exception handling.
- **Graphical User Interface**: Offers code input, token display, AST visualization (both textual and graphical), parse tables, symbol tables, error reporting, and execution results.
- **Extensible Architecture**: Easily extendable to support additional language features.

## Language Syntax

The TEAM language is a statically-typed, imperative programming language with support for object-oriented and functional programming paradigms. Below are the core syntax and features.

### Data Types

- **Primitive Types**:
  - `int`: Integer numbers.
  - `float`: Floating-point numbers.
  - `string`: Textual data.
  - `bool`: Boolean values (`true`, `false`).
  - `void`: Represents no value.

- **Composite Types**:
  - `array`: Fixed-size arrays.
  - `object`: Generic objects.

### Variables and Constants

- **Variable Declaration**:
  ```team
  var x: int;
  let y: float = 3.14;
  const z: string = "Hello World";
  ```

- **Constants**: Declared using `const`, cannot be reassigned after initialization.

### Arrays

- **Declaration and Initialization**:
  ```team
  array arr: int[5] = {1, 2, 3, 4, 5};
  ```

- **Access and Assignment**:
  ```team
  arr[0] = 10;
  print(arr[0]);
  ```

### Functions

- **Declaration**:
  ```team
  func add(a: int, b: int): int {
      return a + b;
  }
  ```

- **Calling Functions**:
  ```team
  var result: int = add(5, 3);
  print(result);
  ```

### Control Structures

- **If-Else Statement**:
  ```team
  if (x > 0) {
      y = y + 1.0;
  } else {
      y = y - 1.0;
  }
  ```

- **While Loop**:
  ```team
  while (x < 10) {
      x = x + 1;
  }
  ```

- **For Loop**:
  ```team
  for (var i: int = 0; i < 5; i = i + 1) {
      arr[i] = add(arr[i], x);
      print(arr[i]);
  }
  ```

### Classes

- **Declaration**:
  ```team
  class Calculator {
      var result: float;

      func multiply(a: float, b: float): float {
          return a * b;
      }
  }
  ```

- **Instantiation and Method Calls**:
  ```team
  var calc = Calculator();
  var product = calc.multiply(2.0, 3.5);
  print(product);
  ```

### Exception Handling

- **Try-Catch Statement**:
  ```team
  try {
      var result: float = Calculator.multiply(2.0, 3.5);
      print(result);
  } catch (e) {
      y = -1.0;
      print(y);
  }
  ```

### Built-in Functions

- `print()`: Outputs data to the execution result panel.
  ```team
  print(x);
  print(y);
  print(z);
  ```

## Installation

### Prerequisites

- **Python 3.x**: Ensure Python is installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

### Clone the Repository

```bash
git clone https://github.com/yourusername/team-language-compiler.git
cd team-language-compiler
```

### Install Dependencies

The project primarily relies on Python's standard library. However, ensure `tkinter` is available.

- **For Windows and macOS**: `tkinter` is usually included with Python.
- **For Linux**: You may need to install it separately.

```bash
# For Debian/Ubuntu-based systems
sudo apt-get install python3-tk
```

## Usage

The TEAM Language Compiler and Interpreter offers both a Command-Line Interface (CLI) and a Graphical User Interface (GUI).

### Running the GUI

1. **Navigate to the Project Directory**:
   ```bash
   cd team-language-compiler
   ```

2. **Run the Compiler**:
   ```bash
   python3 team_compiler.py
   ```

   Replace `team_compiler.py` with the actual filename if different.

3. **Using the GUI**:
   - **Source Code Input**: Write or paste your TEAM code in the "Source Code" section.
   - **Tokenize**: Click the "Tokenize" button to view the list of tokens.
   - **Parse & Analyze**: Click "Parse & Analyze" to generate the AST, parse table, symbol table, and execute the code.
   - **Show Parse Table**: View the parsing table generated from the grammar.
   - **AST Visualization**: Explore both textual and graphical representations of the AST.
   - **Errors and Execution**: Check for semantic errors and view execution results.

### Using the CLI

1. **Prepare Your TEAM Code**: Save your TEAM code in a file, e.g., `program.team`.

2. **Run the Compiler**:
   ```bash
   python3 team_compiler.py
   ```

   The CLI will tokenize, parse, compute FIRST and FOLLOW sets, display the symbol table, and launch the GUI for execution.

## Examples

Below is a sample TEAM program demonstrating various language features:

```team
// Variable and Array Declarations
var x: int = 0;
let y: float = 3.14;
const z: string = "Hello World";
array arr: int[5] = {1, 2, 3, 4, 5};

// If-Else Statement
if (x > 0) {
    y = y + 1.0;
} else {
    y = y - 1.0;
}

// While Loop
while (x < 10) {
    x = x + 1;
}

// For Loop
for (var i: int = 0; i < 5; i = i + 1) {
    // Function Declaration
    func add(a: int, b: int): int {
        return a + b;
    }
    arr[i] = add(arr[i], x);
    print(arr[i]);
}

// Class Declaration
class Calculator {
    var result: float;

    func multiply(a: float, b: float): float {
        return a * b;
    }
}

// Try-Catch Statement
try {
    var result: float = Calculator.multiply(2.0, 3.5);
    print(result);
} catch (e) {
    // Handle error
    y = -1.0;
    print(y);
}

// Print Statement
print(x);
print(y);
print(z);
```

### Expected Output

```
1
2
3
4
5
6
```

## Architecture

### Tokenizer

- **Purpose**: Converts source code into a sequence of tokens based on defined specifications.
- **Implementation**: Utilizes regular expressions to identify different token types like keywords, identifiers, literals, operators, etc.
- **Output**: List of `Token` objects containing type, value, line number, and column.

### Parser

- **Purpose**: Analyzes the token sequence against the grammar to build an Abstract Syntax Tree (AST).
- **Implementation**: Recursive descent parser adhering to the defined grammar rules.
- **Error Handling**: Detects syntax errors and reports them with line and column information.

### Abstract Syntax Tree (AST)

- **Purpose**: Represents the hierarchical structure of the source code.
- **Components**: Various AST node classes like `Program`, `VariableDeclaration`, `FunctionDeclaration`, `IfStatement`, etc.
- **Visualization**: Provides both textual and graphical representations for better understanding.

### Symbol Table

- **Purpose**: Manages symbols (variables, functions, classes) and their attributes across different scopes.
- **Implementation**: Hierarchical symbol tables with support for nested scopes.
- **Features**: Detects semantic errors like undeclared variables, duplicate declarations, and scope violations.

### Backend Execution

- **Purpose**: Interprets the AST to execute TEAM code.
- **Capabilities**:
  - Variable and array manipulation.
  - Function and method calls.
  - Control flow execution (if-else, loops, switch).
  - Exception handling with try-catch blocks.
- **Output**: Execution results and printed statements.

### Graphical User Interface

- **Framework**: Built with Tkinter.
- **Components**:
  - **Source Code Editor**: Input area for writing TEAM code.
  - **Token Display**: Shows the list of tokens generated by the tokenizer.
  - **AST Display**: Textual and graphical visualization of the AST.
  - **Parse Table**: Displays the parsing table based on FIRST and FOLLOW sets.
  - **Symbol Table**: Presents the symbol tables for semantic analysis.
  - **Error Reporting**: Lists semantic and syntactic errors.
  - **Execution Output**: Shows the results of code execution.
- **User Interactions**: Buttons to tokenize, parse & analyze, and show parse tables.

## Error Handling

The TEAM Language Compiler and Interpreter provides comprehensive error detection and reporting mechanisms:

- **Lexical Errors**: Unrecognized symbols or invalid tokens.
- **Syntax Errors**: Violations of grammar rules, such as missing semicolons or mismatched parentheses.
- **Semantic Errors**: Issues like undeclared variables, type mismatches, and scope violations.
- **Runtime Errors**: Errors during execution, such as division by zero or array index out of bounds.

All errors are reported with detailed messages including the type of error, description, and the location (line and column) in the source code.

## Contributing

Contributions are welcome! Whether you're fixing bugs, improving documentation, or adding new features, your help is appreciated.

1. **Fork the Repository**
2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/YourFeature
   ```
3. **Commit Your Changes**
   ```bash
   git commit -m "Add some feature"
   ```
4. **Push to the Branch**
   ```bash
   git push origin feature/YourFeature
   ```
5. **Open a Pull Request**

Please ensure your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the [MIT License](LICENSE).

*Happy Coding with TEAM Language! 🚀*