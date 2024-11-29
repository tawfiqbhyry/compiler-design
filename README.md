# Simple Language Parser

This project implements a simple language parser using Python that can tokenize and parse a small set of control structures, variable declarations, and expressions. The parser is designed to recognize a defined set of keywords, data types, and syntax rules.

## Features

- Tokenization of source code into meaningful tokens
- Parsing control structures (`if`, `while`, etc.)
- Variable declarations with type checking
- Support for basic expressions, boolean operations, and assignments
- Parsing function declarations with parameters

## Components

### Token Specification

The parser defines several token types using regular expressions:

- **KEYWORD**: `if`, `else`, `while`, `for`, `return`, `function`
- **NUMBER**: Integer or decimal numbers
- **STRING**: String literals (e.g., `"Hello, World!"`)
- **BOOLEAN**: Boolean values (`true`, `false`)
- **IDENTIFIER**: Variable names
- **BOOL_OP**: Boolean operators (`and`, `or`, `not`)
- **BOOL_COMP**: Boolean comparisons (`==`, `!=`, `<=`, `>=`, `<`, `>`)
- **ASSIGN**: Assignment operator (`=`)
- **OPERATOR**: Arithmetic operators (`+`, `-`, `*`, `/`, `%`, etc.)
- **PARENTHESIS**: Left `(` and right `)`
- **BRACES**: Left `{` and right `}`
- **SEMICOLON**: `;`
- **COLON**: `:`
- **COMMA**: `,`
- **WHITESPACE**: Spaces and tabs
- **NEWLINE**: Line endings
- **COMMENT**: Single-line comments starting with `//`

### Token Class

The `Token` class represents a token with attributes for its type, value, and position in the source code.

### ParseTreeNode Class

The `ParseTreeNode` class represents nodes in the parse tree, allowing for hierarchical representation of parsed structures.

### Tokenization Function

The `tokenize` function takes source code as input and returns a list of tokens.

### ControlStructureParser Class

This class handles the parsing of tokens into a structured format. Key methods include:

- `parse_variable_declaration()`: Parses variable declarations.
- `parse_if_statement()`: Parses `if` statements, including optional `else` clauses.
- `parse_while_statement()`: Parses `while` statements.
- `parse_function_declaration()`: Parses function declarations with parameters.
- `parse_statement_list()`: Parses a list of statements.
- `parse_expression()`: Parses expressions involving operators and operands.

### Error Handling

The parser raises `ParserError` for any unexpected token encounters, providing useful feedback for debugging.

## Usage

1. Ensure you have Python installed on your machine.
2. Save your code into a file named `x.A` in the same directory as this script.
3. Run the script:

   ```bash
   python parser.py
"# compiler-design" 
