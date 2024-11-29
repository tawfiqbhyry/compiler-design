import re

# Define token types using regular expressions
TOKEN_SPECIFICATION = [
    # Single-line comments
    ('COMMENT', r'//[^\n]*'),
    ('KEYWORD', r'\b(if|else|while|for|return|function)\b'),  # Added 'function'
    # Integer or decimal number
    ('NUMBER', r'\d+(\.\d*)?'),
    ('STRING', r'"[^"]*"'),                                  # String literal
    ('BOOLEAN', r'\b(true|false)\b'),                        # Boolean values
    ('IDENTIFIER', r'[A-Za-z_][A-Za-z_0-9]*'),               # Identifiers
    ('BOOL_OP', r'\b(and|or|not)\b'),                        # Boolean operators
    # Boolean comparisons
    ('BOOL_COMP', r'==|!=|<=|>=|<|>'),
    # Assignment operator
    ('ASSIGN', r'='),
    ('OPERATOR', r'[+\-*/%&|]+'),                            # Operators
    ('LPAREN', r'\('),                                       # Left parenthesis
    # Right parenthesis
    ('RPAREN', r'\)'),
    ('LBRACE', r'\{'),                                       # Left brace
    ('RBRACE', r'\}'),                                       # Right brace
    ('SEMICOLON', r';'),                                     # Semicolon
    ('COLON', r':'),                                         # Colon
    ('COMMA', r','),                                         # Comma
    # Skip over spaces and tabs
    ('WHITESPACE', r'[ \t]+'),
    ('NEWLINE', r'\n'),                                      # Line endings

]

# Compile regular expressions
TOKENS_RE = '|'.join(
    f'(?P<{pair[0]}>{pair[1]})' for pair in TOKEN_SPECIFICATION)


class ParseTreeNode:
    def __init__(self, node_type, value=None):
        self.node_type = node_type  # Type of the node
        self.value = value           # Value of the node
        self.children = []           # Child nodes

    def add_child(self, child_node):
        self.children.append(child_node)

    def __repr__(self):
        return f'{self.node_type}({self.value})'


# Tokenizer
class Token:
    def __init__(self, type, value, line, column):
        self.type = type
        self.value = value
        self.line = line
        self.column = column

    def __repr__(self):
        return f'{self.type}({self.value})'


def tokenize(code):
    line_num = 1
    line_start = 0
    tokens = []
    for match in re.finditer(TOKENS_RE, code):
        kind = match.lastgroup
        value = match.group(kind)
        column = match.start() - line_start
        if kind == 'WHITESPACE' or kind == 'COMMENT':
            continue
        elif kind == 'NEWLINE':
            line_num += 1
            line_start = match.end()
        else:
            tokens.append(Token(kind, value, line_num, column))
    return tokens


# Parser for Control Structures
class ParserError(Exception):
    pass


class ControlStructureParser:
    DATA_TYPES = [
        'int', 'float', 'double', 'char', 'string', 'bool', 'void', 'byte',
        'short', 'long', 'decimal', 'object', 'list', 'set', 'map', 'array',
        'function', 'date', 'time', 'datetime', 'buffer', 'stream', 'enum',
        'struct', 'class', 'interface', 'tuple', 'json', 'xml', 'html',
        'url', 'path', 'regex', 'pointer', 'reference', 'native', 'async',
        'generator', 'promise', 'callback', 'task', 'module'
    ]

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.scope = {}

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def advance(self):
        token = self.peek()
        self.pos += 1
        return token

    def expect(self, token_type):
        token = self.advance()
        if token is None or token.type != token_type:
            raise ParserError(f'Expected {token_type}, got {
                              token.type if token else "EOF"}')
        return token

    def parse_while_statement(self):
        self.expect('KEYWORD')  # "while"
        self.expect('LPAREN')
        self.parse_boolean_expression()
        self.expect('RPAREN')
        self.expect('LBRACE')
        self.parse_statement_list()
        self.expect('RBRACE')

    def parse_if_statement(self):
        self.expect('KEYWORD')  # "if"
        self.expect('LPAREN')
        self.parse_boolean_expression()
        self.expect('RPAREN')
        self.expect('LBRACE')
        self.parse_statement_list()
        self.expect('RBRACE')

        # Check for optional 'else' block
        if self.peek() and self.peek().value == 'else':
            self.advance()  # "else"
            if self.peek() and self.peek().type == 'KEYWORD' and self.peek().value == 'if':
                self.parse_if_statement()  # Else-if (nested if)
            else:
                self.expect('LBRACE')
                self.parse_statement_list()
                self.expect('RBRACE')

    def parse_right_side(self):
        """Parse the right side of a variable declaration into a parse tree."""
        return self.parse_expression()

    def parse_expression(self):
        """Parse an expression and return the root of the parse tree."""
        node = self.parse_term()

        while self.peek() and self.peek().type == 'OPERATOR':
            operator_token = self.advance()
            operator_node = ParseTreeNode('OPERATOR', operator_token.value)
            operator_node.add_child(node)  # Left child
            operator_node.add_child(self.parse_term())  # Right child
            node = operator_node  # Move up the tree

        return node

    def parse_term(self):
        """Parse a term (number, boolean, identifier, etc.) and return a parse tree node."""
        token = self.peek()

        if token.type == 'NUMBER':
            return ParseTreeNode('NUMBER', self.advance().value)
        elif token.type == 'BOOLEAN':
            return ParseTreeNode('BOOLEAN', self.advance().value)
        elif token.type == 'STRING':
            return ParseTreeNode('STRING', self.advance().value)
        elif token.type == 'IDENTIFIER':
            return ParseTreeNode('IDENTIFIER', self.advance().value)
        elif token.type == 'LPAREN':
            self.advance()  # Consume '('
            expr_node = self.parse_expression()
            self.expect('RPAREN')  # Consume ')'
            return expr_node

        raise ParserError(f'Unexpected token {
                          token.value} in right side expression')

    def parse_variable_declaration(self):
        token = self.peek()

        # Check for a data type
        if token.type == 'IDENTIFIER' and token.value in self.DATA_TYPES:
            type_token = self.advance()  # Expect identifier
            var_name = self.expect('IDENTIFIER')  # Expect identifier
            # Store the variable type in the current scope
            self.scope[var_name.value] = type_token.value
        else:
            raise ParserError(f'Expected a data type, got {token.type}')

        # Check for optional assignment
        if self.peek() and self.peek().type == 'ASSIGN':
            self.expect('ASSIGN')  # Consume '='
            initializer = self.parse_right_side()  # Parse the expression for initialization
            self.expect('SEMICOLON')
            print(f"Variable {var_name.value} declared with type {
                type_token.value}")
            return {
                'type': type_token.value,
                'name': var_name.value,
                'initializer': initializer
            }
        else:
            self.expect('SEMICOLON')

        print(f"Variable {var_name.value} declared with type {
              type_token.value}")
        return {
            'type': type_token.value,
            'name': var_name.value,
            'initializer': None
        }

    def parse_boolean_expression(self):
        left = self.parse_relational_expression()

        while self.peek() and (self.peek().type == 'BOOL_OP' or self.peek().type == 'BOOL_COMP'):
            operator = self.advance().value
            right = self.parse_relational_expression()
            left = ParseTreeNode('BOOLEAN_EXPRESSION', operator)
            left.add_child(left)
            left.add_child(right)

        return left

    def parse_relational_expression(self):
        left = self.parse_term()  # Use term for simplicity

        while self.peek() and self.peek().type == 'BOOL_COMP':
            operator = self.advance().value
            right = self.parse_term()
            left = ParseTreeNode('RELATIONAL_EXPRESSION', operator)
            left.add_child(left)
            left.add_child(right)

        return left

    def parse_statement_list(self):
        while self.peek() and self.peek().type != 'RBRACE':
            self.parse_statement()

    def parse_statement(self):
        token = self.peek()
        if token.type == 'COMMENT':
            self.advance()  # Skip comments
        if token.type == 'KEYWORD' and token.value == 'if':
            self.parse_if_statement()
        elif token.type == 'KEYWORD' and token.value == 'while':
            self.parse_while_statement()
        elif token.type == 'IDENTIFIER' and token.value in self.DATA_TYPES:
            self.parse_variable_declaration()
        elif token.type == 'KEYWORD' and token.value == 'function':
            self.parse_function_declaration()
        else:
            raise ParserError(f'Unexpected token {token.value} in statement')

    def parse_function_declaration(self):
        self.expect('KEYWORD')  # "function"
        func_name = self.expect('IDENTIFIER')  # Function name
        self.expect('LPAREN')  # Opening parenthesis

        # Parse parameters
        params = []
        while self.peek() and self.peek().type != 'RPAREN':
            param_type = self.expect('IDENTIFIER')  # Expect a type
            param_name = self.expect('IDENTIFIER')  # Expect a variable name
            params.append((param_type.value, param_name.value))

            if self.peek() and self.peek().type == 'COMMA':
                self.advance()  # Consume comma

        self.expect('RPAREN')  # Closing parenthesis
        self.expect('LBRACE')  # Opening brace for function body
        self.parse_statement_list()  # Parse function body
        self.expect('RBRACE')  # Closing brace

        print(f"Function {func_name.value} declared with parameters: {params}")

    def parse(self):
        while self.peek():
            self.parse_statement()
            


# Example Usage
if __name__ == "__main__":
    code = open('x.A', 'r', encoding="utf-8").read()

    tokens = tokenize(code)
    print("Tokens:")
    print(tokens)

    parser = ControlStructureParser(tokens)
    try:
        parser.parse_statement_list()
        print("Parsing completed successfully.")
    except ParserError as e:
        print(e)
