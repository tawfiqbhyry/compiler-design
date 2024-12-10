import re

# Define token specifications
TOKEN_SPECIFICATION = [
    ('COMMENT', r'//[^\n]*'),                                  # Single-line comments
    ('KEYWORD', r'\b(if|else|while|for|return|function|then|do|break|continue|var|let|const|switch|case|import|export|from|to)\b'),
    ('BOOLEAN', r'\b(true|false)\b'),                         # Boolean values
    ('NUMBER', r'\b\d+(\.\d+)?\b'),                           # Integer or floating-point numbers
    ('STRING', r'"[^"\\]*(?:\\.[^"\\]*)*"'),                  # String literals
    ('IDENTIFIER', r'\b[A-Za-z_][A-Za-z_0-9]*\b'),            # Identifiers
    ('BOOL_COMP', r'==|!=|<=|>=|<|>'),                        # Boolean comparisons
    ('ASSIGN', r'='),                                         # Assignment operator
    ('OPERATOR', r'[+\-*/%&|!]+'),                             # Arithmetic and logical operators
    ('LPAREN', r'\('),                                        # Left parenthesis
    ('RPAREN', r'\)'),                                        # Right parenthesis
    ('LBRACE', r'\{'),                                        # Left brace
    ('RBRACE', r'\}'),                                        # Right brace
    ('SEMICOLON', r';'),                                      # Semicolon
    ('COLON', r':'),                                          # Colon
    ('COMMA', r','),                                          # Comma
    ('WHITESPACE', r'[ \t]+'),                                # Spaces and tabs
    ('NEWLINE', r'\n'),                                       # Line endings
]

# Compile regex
TOKENS_RE = '|'.join(f'(?P<{pair[0]}>{pair[1]})' for pair in TOKEN_SPECIFICATION)
TOKEN_REGEX = re.compile(TOKENS_RE)

# Token class
class Token:
    def __init__(self, type, value, line, column):
        self.type = type
        self.value = value
        self.line = line
        self.column = column
    
    def __repr__(self):
        return f'Token({self.type}, {self.value}, {self.line}, {self.column})'

def tokenize(code):
    line_num = 1
    line_start = 0
    tokens = []
    for mo in TOKEN_REGEX.finditer(code):
        kind = mo.lastgroup
        value = mo.group(kind)
        column = mo.start() - line_start
        if kind == 'WHITESPACE' or kind == 'COMMENT':
            continue
        elif kind == 'NEWLINE':
            line_num += 1
            line_start = mo.end()
            continue
        elif kind == 'STRING':
            value = value[1:-1]  # Remove surrounding quotes
        tokens.append(Token(kind, value, line_num, column))
    return tokens

# AST Node classes
class ASTNode:
    pass

class Program(ASTNode):
    def __init__(self, statements):
        self.statements = statements

class VariableDeclaration(ASTNode):
    def __init__(self, name, initializer):
        self.name = name
        self.initializer = initializer

class FunctionDeclaration(ASTNode):
    def __init__(self, name, params, body):
        self.name = name
        self.params = params
        self.body = body

class IfStatement(ASTNode):
    def __init__(self, condition, then_branch, else_branch):
        self.condition = condition
        self.then_branch = then_branch
        self.else_branch = else_branch

class WhileStatement(ASTNode):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

class ReturnStatement(ASTNode):
    def __init__(self, expression):
        self.expression = expression

class ExpressionStatement(ASTNode):
    def __init__(self, expression):
        self.expression = expression

class Block(ASTNode):
    def __init__(self, statements):
        self.statements = statements

class BinaryExpression(ASTNode):
    def __init__(self, left, operator, right):
        self.left = left
        self.operator = operator
        self.right = right

class UnaryExpression(ASTNode):
    def __init__(self, operator, operand):
        self.operator = operator
        self.operand = operand

class Literal(ASTNode):
    def __init__(self, value):
        self.value = value

class Identifier(ASTNode):
    def __init__(self, name):
        self.name = name

class AssignmentExpression(ASTNode):
    def __init__(self, identifier, value):
        self.identifier = identifier
        self.value = value

class FunctionCall(ASTNode):
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments

# Symbol class with comprehensive attributes
class Symbol:
    def __init__(self, name, kind, data_type=None, scope=None, declared_at=None):
        self.name = name
        self.kind = kind  # 'variable', 'function', or 'parameter'
        self.data_type = data_type  # Optional: can be inferred or set to None
        self.scope = scope  # Scope name where the symbol is declared
        self.declared_at = declared_at  # Tuple (line, column)
    
    def __repr__(self):
        return (f"Symbol(name={self.name}, kind={self.kind}, "
                f"data_type={self.data_type}, scope={self.scope}, "
                f"declared_at={self.declared_at})")

# Symbol Table classes
class SymbolTable:
    def __init__(self, parent=None, scope_name="global"):
        self.symbols = {}
        self.parent = parent
        self.scope_name = scope_name  # e.g., 'global', 'function add', etc.
        self.children = []
        if parent:
            parent.children.append(self)
    
    def define(self, name, kind, data_type=None, declared_at=None):
        if name in self.symbols:
            raise Exception(f"Symbol '{name}' already defined in scope '{self.scope_name}'.")
        symbol = Symbol(name, kind, data_type, scope=self.scope_name, declared_at=declared_at)
        self.symbols[name] = symbol
    
    def lookup(self, name):
        if name in self.symbols:
            return self.symbols[name]
        elif self.parent:
            return self.parent.lookup(name)
        else:
            return None

# Parser implementation using Recursive Descent
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[self.pos] if self.tokens else None
        self.global_symbol_table = SymbolTable(scope_name="global")
        self.symbol_table = self.global_symbol_table
    
    def error(self, message):
        if self.current_token:
            raise Exception(f"Error at line {self.current_token.line}, column {self.current_token.column}: {message}")
        else:
            raise Exception(f"Error at EOF: {message}")
    
    def consume(self, token_type):
        if self.current_token and self.current_token.type == token_type:
            self.advance()
        else:
            expected = token_type
            actual = self.current_token.type if self.current_token else 'EOF'
            self.error(f"Expected token {expected}, got {actual}")
    
    def advance(self):
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = None
    
    def parse(self):
        statements = self.statement_list()
        return Program(statements)
    
    def statement_list(self):
        statements = []
        while self.current_token and self.current_token.type not in ('RBRACE',):
            stmt = self.statement()
            statements.append(stmt)
        return statements
    
    def statement(self):
        if self.current_token.type == 'KEYWORD':
            if self.current_token.value in ('var', 'let', 'const'):
                return self.variable_declaration()
            elif self.current_token.value == 'function':
                return self.function_declaration()
            elif self.current_token.value == 'if':
                return self.if_statement()
            elif self.current_token.value == 'while':
                return self.while_statement()
            elif self.current_token.value == 'return':
                return self.return_statement()
            else:
                return self.expression_statement()
        else:
            return self.expression_statement()
    
    def variable_declaration(self):
        kind = self.current_token.value  # 'var', 'let', or 'const'
        self.consume('KEYWORD')  # Consume 'var', 'let', or 'const'
        if self.current_token.type != 'IDENTIFIER':
            self.error("Expected identifier in variable declaration.")
        var_name = self.current_token.value
        declared_at = (self.current_token.line, self.current_token.column)
        self.symbol_table.define(var_name, 'variable', declared_at=declared_at)
        self.consume('IDENTIFIER')
        initializer = None
        if self.current_token and self.current_token.type == 'ASSIGN':
            self.consume('ASSIGN')
            initializer = self.expression()
        self.consume('SEMICOLON')
        return VariableDeclaration(var_name, initializer)
    
    def function_declaration(self):
        self.consume('KEYWORD')  # 'function'
        if self.current_token.type != 'IDENTIFIER':
            self.error("Expected function name.")
        func_name = self.current_token.value
        declared_at = (self.current_token.line, self.current_token.column)
        self.symbol_table.define(func_name, 'function', declared_at=declared_at)
        self.consume('IDENTIFIER')
        self.consume('LPAREN')
        params = self.parameter_list()
        param_names = [name for name, _ in params]
        self.consume('RPAREN')
        # Define a new scope for the function body
        func_scope_name = f"function {func_name}"
        self.symbol_table = SymbolTable(parent=self.symbol_table, scope_name=func_scope_name)
        # Define parameters in the function's symbol table
        for param_name, param_token in params:
            param_declared_at = (param_token.line, param_token.column)
            self.symbol_table.define(param_name, 'parameter', declared_at=param_declared_at)
        # Parse the function body
        body = self.block(scope_name=func_scope_name)
        # Restore the previous symbol table
        self.symbol_table = self.symbol_table.parent
        return FunctionDeclaration(func_name, param_names, body)
    
    def parameter_list(self):
        params = []
        while self.current_token.type == 'IDENTIFIER':
            param_name = self.current_token.value
            param_token = self.current_token
            params.append((param_name, param_token))
            self.consume('IDENTIFIER')
            if self.current_token.type == 'COMMA':
                self.consume('COMMA')
            else:
                break
        return params
    
    def block(self, scope_name="block"):
        self.consume('LBRACE')
        # Create a new symbol table for the block
        self.symbol_table = SymbolTable(parent=self.symbol_table, scope_name=scope_name)
        statements = self.statement_list()
        self.consume('RBRACE')
        # Restore the previous symbol table
        self.symbol_table = self.symbol_table.parent
        return Block(statements)
    
    def if_statement(self):
        self.consume('KEYWORD')  # 'if'
        self.consume('LPAREN')
        condition = self.expression()
        self.consume('RPAREN')
        then_branch = self.block(scope_name="if")
        else_branch = None
        if self.current_token and self.current_token.type == 'KEYWORD' and self.current_token.value == 'else':
            self.consume('KEYWORD')  # 'else'
            else_branch = self.block(scope_name="else")
        return IfStatement(condition, then_branch, else_branch)
    
    def while_statement(self):
        self.consume('KEYWORD')  # 'while'
        self.consume('LPAREN')
        condition = self.expression()
        self.consume('RPAREN')
        body = self.block(scope_name="while")
        return WhileStatement(condition, body)
    
    def return_statement(self):
        self.consume('KEYWORD')  # 'return'
        expr = None
        if self.current_token.type != 'SEMICOLON':
            expr = self.expression()
        self.consume('SEMICOLON')
        return ReturnStatement(expr)
    
    def expression_statement(self):
        expr = self.expression()
        self.consume('SEMICOLON')
        return ExpressionStatement(expr)
    
    def expression(self):
        return self.assignment()
    
    def assignment(self):
        expr = self.logical_or()
        if self.current_token and self.current_token.type == 'ASSIGN':
            if isinstance(expr, Identifier):
                var_name = expr.name
                self.consume('ASSIGN')
                value = self.assignment()
                # Check if the variable is defined
                symbol = self.symbol_table.lookup(var_name)
                if not symbol:
                    self.error(f"Assignment to undeclared variable '{var_name}'.")
                return AssignmentExpression(var_name, value)
            else:
                self.error("Invalid assignment target.")
        return expr
    
    def logical_or(self):
        expr = self.logical_and()
        while self.current_token and self.current_token.type == 'OPERATOR' and self.current_token.value == '||':
            op = self.current_token.value
            self.consume('OPERATOR')
            right = self.logical_and()
            expr = BinaryExpression(expr, op, right)
        return expr
    
    def logical_and(self):
        expr = self.equality()
        while self.current_token and self.current_token.type == 'OPERATOR' and self.current_token.value == '&&':
            op = self.current_token.value
            self.consume('OPERATOR')
            right = self.equality()
            expr = BinaryExpression(expr, op, right)
        return expr
    
    def equality(self):
        expr = self.relational()
        while self.current_token and self.current_token.type == 'BOOL_COMP' and self.current_token.value in ('==', '!='):
            op = self.current_token.value
            self.consume('BOOL_COMP')
            right = self.relational()
            expr = BinaryExpression(expr, op, right)
        return expr
    
    def relational(self):
        expr = self.additive()
        while self.current_token and self.current_token.type == 'BOOL_COMP' and self.current_token.value in ('<', '>', '<=', '>='):
            op = self.current_token.value
            self.consume('BOOL_COMP')
            right = self.additive()
            expr = BinaryExpression(expr, op, right)
        return expr
    
    def additive(self):
        expr = self.multiplicative()
        while self.current_token and self.current_token.type == 'OPERATOR' and self.current_token.value in ('+', '-'):
            op = self.current_token.value
            self.consume('OPERATOR')
            right = self.multiplicative()
            expr = BinaryExpression(expr, op, right)
        return expr
    
    def multiplicative(self):
        expr = self.unary()
        while self.current_token and self.current_token.type == 'OPERATOR' and self.current_token.value in ('*', '/', '%'):
            op = self.current_token.value
            self.consume('OPERATOR')
            right = self.unary()
            expr = BinaryExpression(expr, op, right)
        return expr
    
    def unary(self):
        if self.current_token and self.current_token.type == 'OPERATOR' and self.current_token.value in ('!', '-'):
            op = self.current_token.value
            self.consume('OPERATOR')
            operand = self.unary()
            return UnaryExpression(op, operand)
        return self.primary()
    
    def primary(self):
        token = self.current_token
        if token.type == 'NUMBER':
            self.consume('NUMBER')
            return Literal(float(token.value) if '.' in token.value else int(token.value))
        elif token.type == 'STRING':
            self.consume('STRING')
            return Literal(token.value)
        elif token.type == 'BOOLEAN':
            self.consume('BOOLEAN')
            return Literal(True if token.value == 'true' else False)
        elif token.type == 'IDENTIFIER':
            if self.peek_next() and self.peek_next().type == 'LPAREN':
                return self.function_call()
            else:
                self.consume('IDENTIFIER')
                return Identifier(token.value)
        elif token.type == 'LPAREN':
            self.consume('LPAREN')
            expr = self.expression()
            self.consume('RPAREN')
            return expr
        else:
            self.error("Unexpected token in expression.")
    
    def function_call(self):
        func_name = self.current_token.value
        self.consume('IDENTIFIER')
        self.consume('LPAREN')
        args = self.argument_list()
        self.consume('RPAREN')
        return FunctionCall(func_name, args)
    
    def argument_list(self):
        args = []
        if self.current_token.type != 'RPAREN':
            args.append(self.expression())
            while self.current_token.type == 'COMMA':
                self.consume('COMMA')
                args.append(self.expression())
        return args
    
    def peek_next(self):
        if self.pos + 1 < len(self.tokens):
            return self.tokens[self.pos + 1]
        else:
            return None

# AST Printer for visualization
def print_ast(node, indent=0):
    prefix = '  ' * indent
    if isinstance(node, Program):
        print(f"{prefix}Program")
        for stmt in node.statements:
            print_ast(stmt, indent + 1)
    elif isinstance(node, VariableDeclaration):
        print(f"{prefix}VariableDeclaration: {node.name}")
        if node.initializer:
            print_ast(node.initializer, indent + 1)
    elif isinstance(node, FunctionDeclaration):
        print(f"{prefix}FunctionDeclaration: {node.name} Params: {node.params}")
        print_ast(node.body, indent + 1)
    elif isinstance(node, IfStatement):
        print(f"{prefix}IfStatement")
        print(f"{prefix}  Condition:")
        print_ast(node.condition, indent + 2)
        print(f"{prefix}  Then:")
        print_ast(node.then_branch, indent + 2)
        if node.else_branch:
            print(f"{prefix}  Else:")
            print_ast(node.else_branch, indent + 2)
    elif isinstance(node, WhileStatement):
        print(f"{prefix}WhileStatement")
        print(f"{prefix}  Condition:")
        print_ast(node.condition, indent + 2)
        print(f"{prefix}  Body:")
        print_ast(node.body, indent + 2)
    elif isinstance(node, ReturnStatement):
        print(f"{prefix}ReturnStatement")
        if node.expression:
            print_ast(node.expression, indent + 1)
    elif isinstance(node, ExpressionStatement):
        print(f"{prefix}ExpressionStatement")
        print_ast(node.expression, indent + 1)
    elif isinstance(node, Block):
        print(f"{prefix}Block")
        for stmt in node.statements:
            print_ast(stmt, indent + 1)
    elif isinstance(node, BinaryExpression):
        print(f"{prefix}BinaryExpression: {node.operator}")
        print_ast(node.left, indent + 1)
        print_ast(node.right, indent + 1)
    elif isinstance(node, UnaryExpression):
        print(f"{prefix}UnaryExpression: {node.operator}")
        print_ast(node.operand, indent + 1)
    elif isinstance(node, Literal):
        print(f"{prefix}Literal: {node.value}")
    elif isinstance(node, Identifier):
        print(f"{prefix}Identifier: {node.name}")
    elif isinstance(node, AssignmentExpression):
        print(f"{prefix}Assignment: {node.identifier} =")
        print_ast(node.value, indent + 1)
    elif isinstance(node, FunctionCall):
        print(f"{prefix}FunctionCall: {node.name}")
        for arg in node.arguments:
            print_ast(arg, indent + 1)
    else:
        print(f"{prefix}Unknown node type: {type(node).__name__}")

# Symbol Table Printer for visualization
def print_symbol_table(symbol_table, indent=0):
    prefix = '  ' * indent
    print(f"{prefix}Scope: {symbol_table.scope_name}")
    for symbol in symbol_table.symbols.values():
        print(f"{prefix}  {symbol.name} : {symbol.kind}, Declared at: Line {symbol.declared_at[0]}, Column {symbol.declared_at[1]}")
    for child in symbol_table.children:
        print_symbol_table(child, indent + 1)

# Example usage of the tokenizer and parser
if __name__ == "__main__":
    sample_code = '''
    // This is a comment
    function add(a, b) {
        var result = a + b;
        return result;
    }

    var i = 0; // Declaration of 'i'

    if (a > b) {
        return a;
    } else {
        return b;
    }

    while (i < 10) {
        i = i + 1;
    }
    '''
    print("Tokenizing...\n")
    tokens = tokenize(sample_code)
    for token in tokens:
        print(token)
    
    print("\nParsing...\n")
    parser = Parser(tokens)
    ast = parser.parse()
    print_ast(ast)
    
    print("\nSymbol Table:\n")
    print_symbol_table(parser.global_symbol_table)
