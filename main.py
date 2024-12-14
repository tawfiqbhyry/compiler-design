import re
import tkinter as tk
from tkinter import scrolledtext

# Define token specifications
TOKEN_SPECIFICATION = [
    ('COMMENT', r'//[^\n]*'),                                 # Single-line comments
    ('KEYWORD', r'\b(if|else|while|for|return|func|then|do|break|continue|var|let|const|switch|case|import|export|from|to|array|class|try|catch|default)\b'),
    ('BOOLEAN', r'\b(true|false)\b'),                         # Boolean values
    ('TYPE', r'\b(int|float|string|bool|void|array|object)\b'),  # Types
    ('NUMBER', r'\b\d+(\.\d+)?\b'),                           # Integer or floating-point numbers
    ('STRING', r'"[^"\\]*(?:\\.[^"\\]*)*"'),                  # String literals
    ('IDENTIFIER', r'\b[A-Za-z_][A-Za-z_0-9]*\b'),            # Identifiers
    ('BOOL_COMP', r'==|!=|<=|>=|<|>'),                        # Boolean comparisons
    ('ASSIGN', r'='),                                         # Assignment operator
    ('OPERATOR', r'[+\-*/%&|!]+'),                            # Arithmetic and logical operators
    ('LPAREN', r'\('),                                        # Left parenthesis
    ('RPAREN', r'\)'),                                        # Right parenthesis
    ('LBRACE', r'\{'),                                        # Left brace
    ('RBRACE', r'\}'),                                        # Right brace
    ('LBRACKET', r'\['),                                      # Left bracket
    ('RBRACKET', r'\]'),                                      # Right bracket
    ('SEMICOLON', r';'),                                      # Semicolon
    ('COLON', r':'),                                          # Colon
    ('COMMA', r','),                                          # Comma
    ('DOT', r'\.'),                                           # Dot
    ('NEWLINE', r'\n'),                                       # Line endings
    ('WHITESPACE', r'[ \t]+'),                                # Spaces and tabs
]

grammar = {
    "Program": [["StatementList"]],
    "MemberFunctionCall": [
        ["IDENTIFIER", "DOT", "IDENTIFIER", "LPAREN", "ArgumentList", "RPAREN"]
    ],
    "Term": [
        ["IDENTIFIER"],
        ["Literal"],
        ["FunctionCall"],
        ["MemberFunctionCall"],  # New addition
        ["ArrayAccess"],
        ["LPAREN", "Expression", "RPAREN"]
    ],
    "StatementList": [["Statement", "StatementList"], ["ε"]],
    "Statement": [
        ["VariableDeclaration"],
        ["ArrayDeclaration"],
        ["AssignmentStatement"],
        ["ArrayAssignment"],
        ["FunctionDeclaration"],
        ["IfStatement"],
        ["WhileStatement"],
        ["ForStatement"],
        ["SwitchStatement"],
        ["ClassDeclaration"],
        ["ReturnStatement"],
        ["TryCatchStatement"],
        ["ExpressionStatement"],
        ["Comment"]
    ],
    "VariableDeclaration": [
        ["var", "IDENTIFIER", "COLON", "TYPE", "SEMICOLON"],
        ["let", "IDENTIFIER", "COLON", "TYPE", "SEMICOLON"],
        ["const", "IDENTIFIER", "COLON", "TYPE", "SEMICOLON"],
        ["var", "IDENTIFIER", "COLON", "TYPE", "ASSIGN", "Expression", "SEMICOLON"],
        ["let", "IDENTIFIER", "COLON", "TYPE", "ASSIGN", "Expression", "SEMICOLON"],
        ["const", "IDENTIFIER", "COLON", "TYPE", "ASSIGN", "Expression", "SEMICOLON"]
    ],
    "ArrayDeclaration": [
        ["array", "IDENTIFIER", "COLON", "TYPE", "LBRACKET", "NUMBER", "RBRACKET", "SEMICOLON"],
        ["array", "IDENTIFIER", "COLON", "TYPE", "LBRACKET", "NUMBER", "RBRACKET", "ASSIGN", "ArrayInitialization", "SEMICOLON"]
    ],
    "ArrayInitialization": [
        ["LBRACE", "ExpressionList", "RBRACE"]
    ],
    "AssignmentStatement": [
        ["IDENTIFIER", "ASSIGN", "Expression", "SEMICOLON"]
    ],
    "ArrayAssignment": [
        ["IDENTIFIER", "LBRACKET", "Expression", "RBRACKET", "ASSIGN", "Expression", "SEMICOLON"]
    ],
    "FunctionDeclaration": [
        ["func", "IDENTIFIER", "LPAREN", "ParameterList", "RPAREN", "COLON", "TYPE", "LBRACE", "StatementList", "RBRACE"]
    ],
    "IfStatement": [
        ["if", "LPAREN", "Expression", "RPAREN", "LBRACE", "StatementList", "RBRACE", "else", "LBRACE", "StatementList", "RBRACE"],
        ["if", "LPAREN", "Expression", "RPAREN", "LBRACE", "StatementList", "RBRACE"]
    ],
    "WhileStatement": [
        ["while", "LPAREN", "Expression", "RPAREN", "LBRACE", "StatementList", "RBRACE"]
    ],
    "ForStatement": [
        ["for", "LPAREN", "AssignmentStatement", "Expression", "SEMICOLON", "Expression", "RPAREN", "LBRACE", "StatementList", "RBRACE"]
    ],
    "SwitchStatement": [
        ["switch", "LPAREN", "Expression", "RPAREN", "LBRACE", "CaseList", "DefaultCase", "RBRACE"],
        ["switch", "LPAREN", "Expression", "RPAREN", "LBRACE", "CaseList", "RBRACE"]
    ],
    "CaseList": [
        ["Case", "CaseList"],
        ["Case"]
    ],
    "Case": [
        ["case", "Expression", "COLON", "StatementList"]
    ],
    "DefaultCase": [
        ["default", "COLON", "StatementList"]
    ],
    "ClassDeclaration": [
        ["class", "IDENTIFIER", "LBRACE", "ClassBody", "RBRACE"]
    ],
    "ClassBody": [
        ["ClassMember", "ClassBody"],
        ["ClassMember"]
    ],
    "ClassMember": [
        ["VariableDeclaration"],
        ["FunctionDeclaration"]
    ],
    "TryCatchStatement": [
        ["try", "LBRACE", "StatementList", "RBRACE", "catch", "LPAREN", "IDENTIFIER", "RPAREN", "LBRACE", "StatementList", "RBRACE"]
    ],
    "ExpressionStatement": [
        ["Expression", "SEMICOLON"]
    ],
    "Expression": [
        ["Expression", "OPERATOR", "Expression"],
        ["Term"]
    ],
    "Term": [
        ["IDENTIFIER"],
        ["Literal"],
        ["FunctionCall"],
        ["ArrayAccess"],
        ["LPAREN", "Expression", "RPAREN"]
    ],
    "FunctionCall": [
        ["IDENTIFIER", "LPAREN", "ArgumentList", "RPAREN"]
    ],
    "ArrayAccess": [
        ["IDENTIFIER", "LBRACKET", "Expression", "RBRACKET"]
    ],
    "ParameterList": [
        ["Parameter", "COMMA", "ParameterList"],
        ["Parameter"]
    ],
    "Parameter": [
        ["IDENTIFIER", "COLON", "TYPE"]
    ],
    "ArgumentList": [
        ["Expression", "COMMA", "ArgumentList"],
        ["Expression"],
        ["ε"]
    ],
    "ExpressionList": [
        ["Expression", "COMMA", "ExpressionList"],
        ["Expression"],
        ["ε"]
    ],
    "Literal": [
        ["NUMBER"],
        ["STRING"],
        ["BOOLEAN"]
    ],
    "Comment": [
        ["COMMENT"]
    ],
    "ε": []
}

def compute_first(grammar):
    first = {non_terminal: set() for non_terminal in grammar}

    def first_of(symbol):
        if symbol in first:  # Non-terminal
            return first[symbol]
        else:  # Terminal
            return {symbol}

    changed = True
    while changed:
        changed = False
        for non_terminal, productions in grammar.items():
            for production in productions:
                for symbol in production:
                    before = len(first[non_terminal])
                    first[non_terminal].update(first_of(symbol))
                    if "ε" not in first_of(symbol):
                        break
                else:
                    first[non_terminal].add("ε")
                if len(first[non_terminal]) > before:
                    changed = True
    return first

def compute_follow(grammar, first):
    follow = {non_terminal: set() for non_terminal in grammar}
    follow["Program"].add("$")  # Assuming Program is the start symbol

    def first_of_sequence(sequence):
        result = set()
        for symbol in sequence:
            result.update(first.get(symbol, {symbol}))
            if "ε" not in first.get(symbol, {symbol}):
                break
        else:
            result.add("ε")
        return result

    changed = True
    while changed:
        changed = False
        for non_terminal, productions in grammar.items():
            for production in productions:
                for i, symbol in enumerate(production):
                    if symbol in grammar:  # Non-terminal
                        next_symbols = production[i + 1:]
                        before = len(follow[symbol])

                        # Add FIRST of next_symbols to FOLLOW of symbol
                        follow[symbol].update(
                            first_of_sequence(next_symbols)
                        )

                        # If next_symbols derive ε, add FOLLOW of current non-terminal
                        if "ε" in first_of_sequence(next_symbols):
                            follow[symbol].update(follow[non_terminal])

                        if len(follow[symbol]) > before:
                            changed = True
    return follow

def display_grammar_and_sets(grammar, first, follow):
    print("Grammar (BNF):")
    for non_terminal, productions in grammar.items():
        productions_str = ' | '.join(' '.join(p) for p in productions if p != ["ε"])
        epsilon = ' | ε' if any(p == ["ε"] for p in productions) else ''
        print(f"{non_terminal} ::= {productions_str}{epsilon}")

    print("\nFIRST Sets:")
    for non_terminal, first_set in first.items():
        print(f"FIRST({non_terminal}) = {{ {', '.join(first_set)} }}")

    print("\nFOLLOW Sets:")
    for non_terminal, follow_set in follow.items():
        print(f"FOLLOW({non_terminal}) = {{ {', '.join(follow_set)} }}")

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
            pass  # Ignore whitespace and comments
        elif kind == 'NEWLINE':
            line_num += 1
            line_start = mo.end()
        elif kind == 'STRING':
            value = value[1:-1]  # Remove surrounding quotes
            tokens.append(Token(kind, value, line_num, column))
        elif kind == 'NUMBER':
            tokens.append(Token(kind, float(value) if '.' in value else int(value), line_num, column))
        else:
            tokens.append(Token(kind, value, line_num, column))
    return tokens

# AST Node classes
class ASTNode:
    pass

class Program(ASTNode):
    def __init__(self, statements):
        self.statements = statements

class VariableDeclaration(ASTNode):
    def __init__(self, kind, name, var_type, initializer=None):
        self.kind = kind  # 'var', 'let', or 'const'
        self.name = name
        self.var_type = var_type
        self.initializer = initializer

class ArrayDeclaration(ASTNode):
    def __init__(self, name, var_type, size, initializer=None):
        self.name = name
        self.var_type = var_type
        self.size = size
        self.initializer = initializer

class AssignmentStatement(ASTNode):
    def __init__(self, identifier, value):
        self.identifier = identifier
        self.value = value

class ArrayAssignment(ASTNode):
    def __init__(self, identifier, index, value):
        self.identifier = identifier
        self.index = index
        self.value = value

class FunctionDeclaration(ASTNode):
    def __init__(self, name, params, return_type, body):
        self.name = name
        self.params = params  # List of tuples (name, type)
        self.return_type = return_type
        self.body = body

class IfStatement(ASTNode):
    def __init__(self, condition, then_branch, else_branch=None):
        self.condition = condition
        self.then_branch = then_branch
        self.else_branch = else_branch

class WhileStatement(ASTNode):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

class ForStatement(ASTNode):
    def __init__(self, init, condition, increment, body):
        self.init = init
        self.condition = condition
        self.increment = increment
        self.body = body

class SwitchStatement(ASTNode):
    def __init__(self, expression, cases, default=None):
        self.expression = expression
        self.cases = cases  # List of tuples (expression, statements)
        self.default = default  # List of statements

class Case(ASTNode):
    def __init__(self, expression, statements):
        self.expression = expression
        self.statements = statements

class DefaultCase(ASTNode):
    def __init__(self, statements):
        self.statements = statements

class ClassDeclaration(ASTNode):
    def __init__(self, name, members):
        self.name = name
        self.members = members  # List of VariableDeclaration or FunctionDeclaration

class TryCatchStatement(ASTNode):
    def __init__(self, try_block, catch_identifier, catch_block):
        self.try_block = try_block
        self.catch_identifier = catch_identifier
        self.catch_block = catch_block

class ReturnStatement(ASTNode):
    def __init__(self, expression=None):
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

class ArrayAccess(ASTNode):
    def __init__(self, name, index):
        self.name = name
        self.index = index

# Symbol class with comprehensive attributes
# Symbol class with comprehensive attributes
class Symbol:
    def __init__(self, name, kind, data_type=None, scope=None, declared_at=None):
        self.name = name
        self.kind = kind  # 'variable', 'function', 'parameter', 'class'
        self.data_type = data_type  # Optional: can be inferred or set to None
        self.scope = scope  # Scope name where the symbol is declared
        self.declared_at = declared_at  # Tuple (line, column)
        self.class_symbol_table = None  # For classes, to store their members
    
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
            return False  # Symbol already defined in this scope
        symbol = Symbol(name, kind, data_type, scope=self.scope_name, declared_at=declared_at)
        self.symbols[name] = symbol
        return True

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
        self.errors = []  # List to collect semantic errors

    def error(self, message):
        if self.current_token:
            error_msg = f"Error at line {self.current_token.line}, column {self.current_token.column}: {message}"
        else:
            error_msg = f"Error at EOF: {message}"
        self.errors.append(error_msg)

    def consume(self, token_type):
        if self.current_token and self.current_token.type == token_type:
            self.advance()
            return True
        else:
            expected = token_type
            actual = self.current_token.type if self.current_token else 'EOF'
            self.error(f"Expected token {expected}, got {actual}")
            return False

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
            if stmt:
                statements.append(stmt)
        return statements

    def statement(self):
        if self.current_token.type == 'KEYWORD':
            keyword = self.current_token.value
            if keyword in ('var', 'let', 'const'):
                return self.variable_declaration()
            elif keyword == 'array':
                return self.array_declaration()
            elif keyword == 'func':
                return self.function_declaration()
            elif keyword == 'if':
                return self.if_statement()
            elif keyword == 'while':
                return self.while_statement()
            elif keyword == 'for':
                return self.for_statement()
            elif keyword == 'switch':
                return self.switch_statement()
            elif keyword == 'class':
                return self.class_declaration()
            elif keyword == 'return':
                return self.return_statement()
            elif keyword == 'try':
                return self.try_catch_statement()
            else:
                return self.expression_statement()
        elif self.current_token.type == 'IDENTIFIER':
            # Could be assignment or function call or member function call
            next_token = self.peek_next()
            if next_token and next_token.type == 'ASSIGN':
                return self.assignment_statement()
            elif next_token and next_token.type == 'LBRACKET':
                return self.array_assignment()
            elif next_token and next_token.type == 'DOT':
                return self.expression_statement()  # Handle member function call within expression
            else:
                return self.expression_statement()
        else:
            return self.expression_statement()

    def variable_declaration(self):
        kind = self.current_token.value  # 'var', 'let', or 'const'
        self.consume('KEYWORD')  # Consume 'var', 'let', or 'const'
        if self.current_token and self.current_token.type == 'IDENTIFIER':
            var_name = self.current_token.value
            self.consume('IDENTIFIER')
            if not self.consume('COLON'):
                return None
            if self.current_token and self.current_token.type == 'TYPE':
                var_type = self.current_token.value
                self.consume('TYPE')
            else:
                var_type = None
                self.error("Expected type in variable declaration.")
            initializer = None
            if self.current_token and self.current_token.type == 'ASSIGN':
                self.consume('ASSIGN')
                initializer = self.expression()
            if not self.consume('SEMICOLON'):
                return None
            # Attempt to define the variable in the current symbol table
            declared_at = (self.current_token.line, self.current_token.column) if self.current_token else (0, 0)
            if not self.symbol_table.define(var_name, 'variable', data_type=var_type, declared_at=declared_at):
                self.error(f"Variable '{var_name}' is already declared in scope '{self.symbol_table.scope_name}'.")
            return VariableDeclaration(kind, var_name, var_type, initializer)
        else:
            self.error("Expected identifier in variable declaration.")
            return None

    def array_declaration(self):
        self.consume('KEYWORD')  # 'array'
        if self.current_token and self.current_token.type == 'IDENTIFIER':
            array_name = self.current_token.value
            self.consume('IDENTIFIER')
            if not self.consume('COLON'):
                return None
            if self.current_token and self.current_token.type == 'TYPE':
                array_type = self.current_token.value
                self.consume('TYPE')
            else:
                array_type = None
                self.error("Expected type in array declaration.")
            if not self.consume('LBRACKET'):
                return None
            if self.current_token and self.current_token.type == 'NUMBER':
                size = self.current_token.value
                self.consume('NUMBER')
            else:
                size = None
                self.error("Expected size in array declaration.")
            if not self.consume('RBRACKET'):
                return None
            initializer = None
            if self.current_token and self.current_token.type == 'ASSIGN':
                self.consume('ASSIGN')
                initializer = self.array_initialization()
            if not self.consume('SEMICOLON'):
                return None
            # Attempt to define the array in the current symbol table
            declared_at = (self.current_token.line, self.current_token.column) if self.current_token else (0, 0)
            if not self.symbol_table.define(array_name, 'array', data_type=array_type, declared_at=declared_at):
                self.error(f"Array '{array_name}' is already declared in scope '{self.symbol_table.scope_name}'.")
            return ArrayDeclaration(array_name, array_type, size, initializer)
        else:
            self.error("Expected identifier in array declaration.")
            return None

    def array_initialization(self):
        if not self.consume('LBRACE'):
            return None
        expressions = self.expression_list()
        if not self.consume('RBRACE'):
            return None
        return expressions

    def assignment_statement(self):
        if self.current_token.type == 'IDENTIFIER':
            identifier = self.current_token.value
            self.consume('IDENTIFIER')
            if not self.consume('ASSIGN'):
                return None
            value = self.expression()
            if not self.consume('SEMICOLON'):
                return None
            # Check if the identifier is defined
            symbol = self.symbol_table.lookup(identifier)
            if not symbol:
                self.error(f"Assignment to undeclared variable '{identifier}'.")
            return AssignmentStatement(identifier, value)
        else:
            self.error("Expected identifier in assignment.")
            return None

    def array_assignment(self):
        if self.current_token.type == 'IDENTIFIER':
            array_name = self.current_token.value
            self.consume('IDENTIFIER')
            if not self.consume('LBRACKET'):
                return None
            index = self.expression()
            if not self.consume('RBRACKET'):
                return None
            if not self.consume('ASSIGN'):
                return None
            value = self.expression()
            if not self.consume('SEMICOLON'):
                return None
            # Check if the array is defined
            symbol = self.symbol_table.lookup(array_name)
            if not symbol:
                self.error(f"Assignment to undeclared array '{array_name}'.")
            return ArrayAssignment(array_name, index, value)
        else:
            self.error("Expected identifier in array assignment.")
            return None

    def function_declaration(self):
        self.consume('KEYWORD')  # 'func'
        if self.current_token and self.current_token.type == 'IDENTIFIER':
            func_name = self.current_token.value
            func_decl_token = self.current_token  # Capture the token for position
            self.consume('IDENTIFIER')
            if not self.consume('LPAREN'):
                return None
            params = self.parameter_list()
            if not self.consume('RPAREN'):
                return None
            if not self.consume('COLON'):
                return None
            if self.current_token and self.current_token.type == 'TYPE':
                return_type = self.current_token.value
                self.consume('TYPE')
            else:
                return_type = None
                self.error("Expected return type in function declaration.")
            if not self.consume('LBRACE'):
                return None
            
            # **Define the function in the current symbol table before parsing the body**
            declared_at = (func_decl_token.line, func_decl_token.column)
            if not self.symbol_table.define(func_name, 'function', data_type=return_type, declared_at=declared_at):
                self.error(f"Function '{func_name}' is already declared in scope '{self.symbol_table.scope_name}'.")
            
            # Define a new scope for the function body
            func_scope_name = f"function {func_name}"
            self.symbol_table = SymbolTable(parent=self.symbol_table, scope_name=func_scope_name)
            
            # Define parameters in the function's symbol table
            for param_name, param_type in params:
                if not self.symbol_table.define(param_name, 'parameter', data_type=param_type, declared_at=(self.current_token.line, self.current_token.column)):
                    self.error(f"Parameter '{param_name}' is already declared in scope '{self.symbol_table.scope_name}'.")
            
            # Parse the function body
            body = self.statement_list()
            if not self.consume('RBRACE'):
                return None
            
            # Restore the previous symbol table
            self.symbol_table = self.symbol_table.parent
            return FunctionDeclaration(func_name, params, return_type, body)
        else:
            self.error("Expected function name.")
            return None


    def parameter_list(self):
        params = []
        if self.current_token and self.current_token.type == 'IDENTIFIER':
            param = self.parameter()
            params.append(param)
            while self.current_token and self.current_token.type == 'COMMA':
                self.consume('COMMA')
                param = self.parameter()
                params.append(param)
        return params

    def parameter(self):
        if self.current_token.type == 'IDENTIFIER':
            param_name = self.current_token.value
            self.consume('IDENTIFIER')
            if not self.consume('COLON'):
                return (param_name, None)
            if self.current_token and self.current_token.type == 'TYPE':
                param_type = self.current_token.value
                self.consume('TYPE')
                return (param_name, param_type)
            else:
                self.error("Expected type in parameter.")
                return (param_name, None)
        else:
            self.error("Expected identifier in parameter.")
            return (None, None)

    def block(self):
        if not self.consume('LBRACE'):
            return None
        # Create a new symbol table for the block
        self.symbol_table = SymbolTable(parent=self.symbol_table, scope_name="block")
        statements = self.statement_list()
        if not self.consume('RBRACE'):
            return None
        # Restore the previous symbol table
        self.symbol_table = self.symbol_table.parent
        return Block(statements)

    def if_statement(self):
        self.consume('KEYWORD')  # 'if'
        if not self.consume('LPAREN'):
            return None
        condition = self.expression()
        if not self.consume('RPAREN'):
            return None
        if not self.consume('LBRACE'):
            return None
        then_branch = self.statement_list()
        if not self.consume('RBRACE'):
            return None
        else_branch = None
        if self.current_token and self.current_token.type == 'KEYWORD' and self.current_token.value == 'else':
            self.consume('KEYWORD')  # 'else'
            if not self.consume('LBRACE'):
                return None
            else_branch = self.statement_list()
            if not self.consume('RBRACE'):
                return None
        return IfStatement(condition, then_branch, else_branch)

    def while_statement(self):
        self.consume('KEYWORD')  # 'while'
        if not self.consume('LPAREN'):
            return None
        condition = self.expression()
        if not self.consume('RPAREN'):
            return None
        if not self.consume('LBRACE'):
            return None
        body = self.statement_list()
        if not self.consume('RBRACE'):
            return None
        return WhileStatement(condition, body)

    def for_statement(self):
        self.consume('KEYWORD')  # 'for'
        if not self.consume('LPAREN'):
            return None

        if self.current_token and self.current_token.type == 'KEYWORD' and self.current_token.value in ('var', 'let', 'const'):
            init = self.variable_declaration()
        elif self.current_token and self.current_token.type == 'IDENTIFIER':
            init = self.assignment_statement()
        else:
            self.error("Expected VariableDeclaration or AssignmentStatement in for loop initialization.")
            return None
        
        condition = self.expression()
        
        if not self.consume('SEMICOLON'):
            return None
        
        increment = self.expression()
        
        if not self.consume('RPAREN'):
            return None
        
        if not self.consume('LBRACE'):
            return None
        
        body = self.statement_list()
        
        if not self.consume('RBRACE'):
            return None
        
        return ForStatement(init, condition, increment, body)

    def switch_statement(self):
        self.consume('KEYWORD')  # 'switch'
        if not self.consume('LPAREN'):
            return None
        expression = self.expression()
        if not self.consume('RPAREN'):
            return None
        if not self.consume('LBRACE'):
            return None
        cases = []
        default = None
        while self.current_token and self.current_token.type != 'RBRACE':
            if self.current_token.type == 'KEYWORD' and self.current_token.value == 'case':
                case = self.case_statement()
                if case:
                    cases.append(case)
            elif self.current_token.type == 'KEYWORD' and self.current_token.value == 'default':
                default = self.default_case()
            else:
                self.error("Expected 'case' or 'default' in switch statement.")
                break
        if not self.consume('RBRACE'):
            return None
        return SwitchStatement(expression, cases, default)

    def case_statement(self):
        self.consume('KEYWORD')  # 'case'
        expr = self.expression()
        if not self.consume('COLON'):
            return None
        statements = self.statement_list()
        return Case(expr, statements)

    def default_case(self):
        self.consume('KEYWORD')  # 'default'
        if not self.consume('COLON'):
            return None
        statements = self.statement_list()
        return DefaultCase(statements)

    def class_declaration(self):
        self.consume('KEYWORD')  # 'class'
        if self.current_token and self.current_token.type == 'IDENTIFIER':
            class_name = self.current_token.value
            class_decl_token = self.current_token  # Capture the token for position
            self.consume('IDENTIFIER')
            if not self.consume('LBRACE'):
                return None
            members = self.class_body()
            if not self.consume('RBRACE'):
                return None
            # Define the class in the current symbol table
            declared_at = (class_decl_token.line, class_decl_token.column)
            if not self.symbol_table.define(class_name, 'class', declared_at=declared_at):
                self.error(f"Class '{class_name}' is already declared in scope '{self.symbol_table.scope_name}'.")
            # Retrieve the class symbol and create its symbol table
            class_symbol = self.symbol_table.lookup(class_name)
            class_symbol.class_symbol_table = SymbolTable(parent=None, scope_name=f"class {class_name}")
            # Populate the class's symbol table with its members
            for member in members:
                if isinstance(member, FunctionDeclaration):
                    # Define method in class's symbol table
                    if not class_symbol.class_symbol_table.define(member.name, 'function', data_type=member.return_type, declared_at=(0,0)):
                        self.error(f"Method '{member.name}' is already declared in class '{class_name}'.")
                elif isinstance(member, VariableDeclaration):
                    if not class_symbol.class_symbol_table.define(member.name, 'variable', data_type=member.var_type, declared_at=(0,0)):
                        self.error(f"Variable '{member.name}' is already declared in class '{class_name}'.")
            return ClassDeclaration(class_name, members)
        else:
            self.error("Expected class name.")
            return None


    def class_body(self):
        members = []
        while self.current_token and self.current_token.type != 'RBRACE':
            member = self.class_member()
            if member:
                members.append(member)
        return members

    def class_member(self):
        if self.current_token.type == 'KEYWORD':
            keyword = self.current_token.value
            if keyword in ('var', 'let', 'const'):
                return self.variable_declaration()
            elif keyword == 'func':
                return self.function_declaration()
            else:
                self.error(f"Unexpected keyword '{keyword}' in class body.")
                self.consume(self.current_token.type)
                return None
        else:
            self.error("Expected member declaration in class.")
            return None

    def try_catch_statement(self):
        self.consume('KEYWORD')  # 'try'
        if not self.consume('LBRACE'):
            return None
        # **Create a new scope for the try block**
        self.symbol_table = SymbolTable(parent=self.symbol_table, scope_name="try")
        try_block = self.statement_list()
        if not self.consume('RBRACE'):
            return None
        if not (self.current_token and self.current_token.type == 'KEYWORD' and self.current_token.value == 'catch'):
            self.error("Expected 'catch' after 'try' block.")
            return None
        self.consume('KEYWORD')  # 'catch'
        if not self.consume('LPAREN'):
            return None
        if self.current_token and self.current_token.type == 'IDENTIFIER':
            catch_identifier = self.current_token.value
            declared_at = (self.current_token.line, self.current_token.column)  # Capture before consuming
            self.consume('IDENTIFIER')
        else:
            catch_identifier = None
            declared_at = (0, 0)  # Default or previous token's position can be used
            self.error("Expected identifier in catch clause.")
        if not self.consume('RPAREN'):
            return None
        if not self.consume('LBRACE'):
            return None
        # **Create a new scope for the catch block**
        self.symbol_table = SymbolTable(parent=self.symbol_table, scope_name="catch")
        # Define the catch identifier in the symbol table if it exists
        if catch_identifier:
            if not self.symbol_table.define(catch_identifier, 'parameter', data_type=None, declared_at=declared_at):
                self.error(f"Identifier '{catch_identifier}' is already declared in scope '{self.symbol_table.scope_name}'.")
        catch_block = self.statement_list()
        if not self.consume('RBRACE'):
            return None
        # Restore the previous symbol table
        self.symbol_table = self.symbol_table.parent.parent  # Exit 'catch' and 'try' scopes
        return TryCatchStatement(try_block, catch_identifier, catch_block)


    def return_statement(self):
        self.consume('KEYWORD')  # 'return'
        expr = None
        if self.current_token and self.current_token.type != 'SEMICOLON':
            expr = self.expression()
        if not self.consume('SEMICOLON'):
            return None
        return ReturnStatement(expr)

    def expression_statement(self):
        expr = self.expression()
        if not self.consume('SEMICOLON'):
            return None
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
        if not token:
            self.error("Unexpected end of input.")
            return None
        if token.type == 'NUMBER':
            self.consume('NUMBER')
            return Literal(token.value)
        elif token.type == 'STRING':
            self.consume('STRING')
            return Literal(token.value)
        elif token.type == 'BOOLEAN':
            self.consume('BOOLEAN')
            return Literal(True if token.value == 'true' else False)
        elif token.type == 'IDENTIFIER':
            next_token = self.peek_next()
            if next_token and next_token.type == 'LPAREN':
                return self.function_call()
            elif next_token and next_token.type == 'DOT':
                return self.member_function_call()  # Handle member function call
            elif next_token and next_token.type == 'LBRACKET':
                return self.array_access()
            else:
                identifier_name = token.value
                # Check if the identifier is defined
                symbol = self.symbol_table.lookup(identifier_name)
                if not symbol:
                    self.error(f"Use of undeclared identifier '{identifier_name}'.")
                self.consume('IDENTIFIER')
                return Identifier(identifier_name)
        elif token.type == 'LPAREN':
            self.consume('LPAREN')
            expr = self.expression()
            if not self.consume('RPAREN'):
                return None
            return expr
        else:
            self.error("Unexpected token in expression.")
            self.consume(token.type)  # Attempt to recover
            return None


    def function_call(self):
        func_name = self.current_token.value
        self.consume('IDENTIFIER')
        if not self.consume('LPAREN'):
            return None
        args = self.argument_list()
        if not self.consume('RPAREN'):
            return None
        # Check if the function is defined
        symbol = self.symbol_table.lookup(func_name)
        if not symbol:
            self.error(f"Call to undeclared function '{func_name}'.")
        elif symbol.kind != 'function':
            self.error(f"Identifier '{func_name}' is not a function.")
        return FunctionCall(func_name, args)


    def array_access(self):
        array_name = self.current_token.value
        self.consume('IDENTIFIER')
        if not self.consume('LBRACKET'):
            return None
        index = self.expression()
        if not self.consume('RBRACKET'):
            return None
        # Check if the array is defined
        symbol = self.symbol_table.lookup(array_name)
        if not symbol:
            self.error(f"Access to undeclared array '{array_name}'.")
        return ArrayAccess(array_name, index)

    def argument_list(self):
        args = []
        if self.current_token and self.current_token.type != 'RPAREN':
            args.append(self.expression())
            while self.current_token and self.current_token.type == 'COMMA':
                self.consume('COMMA')
                args.append(self.expression())
        return args

    def expression_list(self):
        exprs = []
        if self.current_token and self.current_token.type != 'RBRACE':
            exprs.append(self.expression())
            while self.current_token and self.current_token.type == 'COMMA':
                self.consume('COMMA')
                exprs.append(self.expression())
        return exprs

    def peek_next(self):
        if self.pos + 1 < len(self.tokens):
            return self.tokens[self.pos + 1]
        else:
            return None

    def member_function_call(self):
        object_name = self.current_token.value
        self.consume('IDENTIFIER')
        if not self.consume('DOT'):
            return None
        func_name = self.current_token.value
        self.consume('IDENTIFIER')
        if not self.consume('LPAREN'):
            return None
        args = self.argument_list()
        if not self.consume('RPAREN'):
            return None
        # Check if the object is defined and has the method
        obj_symbol = self.symbol_table.lookup(object_name)
        if not obj_symbol:
            self.error(f"Call to method '{func_name}' on undeclared object '{object_name}'.")
        elif obj_symbol.kind != 'class':
            self.error(f"Identifier '{object_name}' is not a class.")
        else:
            # Look up the method within the class's symbol table
            class_symbol_table = obj_symbol.class_symbol_table  # Assuming class symbols have their own symbol tables
            method_symbol = class_symbol_table.lookup(func_name) if class_symbol_table else None
            if not method_symbol:
                self.error(f"Call to undeclared method '{func_name}' of class '{object_name}'.")
            elif method_symbol.kind != 'function':
                self.error(f"Identifier '{func_name}' is not a function in class '{object_name}'.")
        return FunctionCall(f"{object_name}.{func_name}", args)


# AST Printer for visualization
def print_ast(node, indent=0):
    prefix = '  ' * indent
    result = ""

    if isinstance(node, Program):
        result += f"{prefix}Program\n"
        for stmt in node.statements:
            result += print_ast(stmt, indent + 1)
        return result
    elif isinstance(node, VariableDeclaration):
        result += f"{prefix}VariableDeclaration ({node.kind}) : {node.name} : {node.var_type}\n"
        if node.initializer:
            result += print_ast(node.initializer, indent + 1)
        return result
    elif isinstance(node, ArrayDeclaration):
        result += f"{prefix}ArrayDeclaration: {node.name} : {node.var_type}[{node.size}]\n"
        if node.initializer:
            result += print_ast(node.initializer, indent + 1)
        return result
    elif isinstance(node, AssignmentStatement):
        result += f"{prefix}Assignment: {node.identifier} =\n"
        result += print_ast(node.value, indent + 1)
        return result
    elif isinstance(node, ArrayAssignment):
        result += f"{prefix}ArrayAssignment: {node.identifier}[\n"
        result += print_ast(node.index, indent + 2)
        result += f"{prefix}] =\n"
        result += print_ast(node.value, indent + 2)
        return result
    elif isinstance(node, FunctionDeclaration):
        params_str = ', '.join(f"{name}:{ptype}" for name, ptype in node.params)
        result += f"{prefix}FunctionDeclaration: {node.name}({params_str}) : {node.return_type}\n"
        result += print_ast(node.body, indent + 1)
        return result
    elif isinstance(node, IfStatement):
        result += f"{prefix}IfStatement\n"
        result += f"{prefix}  Condition:\n"
        result += print_ast(node.condition, indent + 2)
        result += f"{prefix}  Then:\n"
        result += print_ast(node.then_branch, indent + 2)
        if node.else_branch:
            result += f"{prefix}  Else:\n"
            result += print_ast(node.else_branch, indent + 2)
        return result
    elif isinstance(node, WhileStatement):
        result += f"{prefix}WhileStatement\n"
        result += f"{prefix}  Condition:\n"
        result += print_ast(node.condition, indent + 2)
        result += f"{prefix}  Body:\n"
        result += print_ast(node.body, indent + 2)
        return result
    elif isinstance(node, ForStatement):
        result += f"{prefix}ForStatement\n"
        result += f"{prefix}  Init:\n"
        result += print_ast(node.init, indent + 2)
        result += f"{prefix}  Condition:\n"
        result += print_ast(node.condition, indent + 2)
        result += f"{prefix}  Increment:\n"
        result += print_ast(node.increment, indent + 2)
        result += f"{prefix}  Body:\n"
        result += print_ast(node.body, indent + 2)
        return result
    elif isinstance(node, SwitchStatement):
        result += f"{prefix}SwitchStatement\n"
        result += f"{prefix}  Expression:\n"
        result += print_ast(node.expression, indent + 2)
        for case in node.cases:
            result += print_ast(case, indent + 1)
        if node.default:
            result += print_ast(node.default, indent + 1)
        return result
    elif isinstance(node, Case):
        result += f"{prefix}Case:\n"
        result += print_ast(node.expression, indent + 2)
        result += f"{prefix}  Statements:\n"
        result += print_ast(node.statements, indent + 2)
        return result
    elif isinstance(node, DefaultCase):
        result += f"{prefix}DefaultCase:\n"
        result += print_ast(node.statements, indent + 2)
        return result
    elif isinstance(node, ClassDeclaration):
        result += f"{prefix}ClassDeclaration: {node.name}\n"
        for member in node.members:
            result += print_ast(member, indent + 1)
        return result
    elif isinstance(node, TryCatchStatement):
        result += f"{prefix}TryCatchStatement\n"
        result += f"{prefix}  Try Block:\n"
        result += print_ast(node.try_block, indent + 2)
        result += f"{prefix}  Catch Identifier: {node.catch_identifier}\n"
        result += f"{prefix}  Catch Block:\n"
        result += print_ast(node.catch_block, indent + 2)
        return result
    elif isinstance(node, ReturnStatement):
        result += f"{prefix}ReturnStatement\n"
        if node.expression:
            result += print_ast(node.expression, indent + 1)
        return result
    elif isinstance(node, ExpressionStatement):
        result += f"{prefix}ExpressionStatement\n"
        result += print_ast(node.expression, indent + 1)
        return result
    elif isinstance(node, Block):
        result += f"{prefix}Block\n"
        for stmt in node.statements:
            result += print_ast(stmt, indent + 1)
        return result
    elif isinstance(node, BinaryExpression):
        result += f"{prefix}BinaryExpression: {node.operator}\n"
        result += print_ast(node.left, indent + 1)
        result += print_ast(node.right, indent + 1)
        return result
    elif isinstance(node, UnaryExpression):
        result += f"{prefix}UnaryExpression: {node.operator}\n"
        result += print_ast(node.operand, indent + 1)
        return result
    elif isinstance(node, Literal):
        result += f"{prefix}Literal: {node.value}\n"
        return result
    elif isinstance(node, Identifier):
        result += f"{prefix}Identifier: {node.name}\n"
        return result
    elif isinstance(node, AssignmentExpression):
        result += f"{prefix}AssignmentExpression: {node.identifier} =\n"
        result += print_ast(node.value, indent + 1)
        return result
    elif isinstance(node, FunctionCall):
        result += f"{prefix}FunctionCall: {node.name}\n"
        for arg in node.arguments:
            result += print_ast(arg, indent + 1)
        return result
    elif isinstance(node, ArrayAccess):
        result += f"{prefix}ArrayAccess: {node.name}[\n"
        result += print_ast(node.index, indent + 2)
        result += f"{prefix}]\n"
        return result
    else:
        result += f"{prefix}Unknown node type: {type(node).__name__}\n"
        return result

# Symbol Table Printer for visualization
def print_symbol_table(symbol_table, indent=0):
    prefix = '  ' * indent
    result = f"{prefix}Scope: {symbol_table.scope_name}\n"
    for symbol in symbol_table.symbols.values():
        result += f"{prefix}  {symbol.name} : {symbol.kind}, Type: {symbol.data_type}, Declared at: Line {symbol.declared_at[0]}, Column {symbol.declared_at[1]}\n"
    for child in symbol_table.children:
        result += print_symbol_table(child, indent + 1)
    return result

def create_parse_table(grammar, first, follow):
    parse_table = {}
    for non_terminal in grammar:
        parse_table[non_terminal] = {}
        for production in grammar[non_terminal]:
            first_seq = first_of_sequence(production, first)
            for terminal in first_seq:
                if terminal != "ε":
                    if terminal in parse_table[non_terminal]:
                        parse_table[non_terminal][terminal].append(production)
                    else:
                        parse_table[non_terminal][terminal] = [production]
            if "ε" in first_of_sequence(production, first):
                for terminal in follow[non_terminal]:
                    if terminal in parse_table[non_terminal]:
                        parse_table[non_terminal][terminal].append(production)
                    else:
                        parse_table[non_terminal][terminal] = [production]
    return parse_table

def first_of_sequence(sequence, first):
    result = set()
    for symbol in sequence:
        result.update(first.get(symbol, {symbol}))
        if "ε" not in first.get(symbol, {symbol}):
            break
    else:
        result.add("ε")
    return result

def display_parse_table(parse_table):
    print("Parse Table:")
    for non_terminal, row in parse_table.items():
        print(f"{non_terminal}:")
        for terminal, productions in row.items():
            for production in productions:
                prod_str = ' '.join(production)
                print(f"  {terminal} -> {prod_str}")
def GUI():
    def tokenize_input():
        code = input_text.get("1.0", tk.END).strip()
        tokens = tokenize(code)
        token_output.delete("1.0", tk.END)
        for token in tokens:
            token_output.insert(tk.END, f"{token}\n")

    def parse_input():
        code = input_text.get("1.0", tk.END).strip()
        tokens = tokenize(code)
        parser = Parser(tokens)
        ast = parser.parse()
        # Collect semantic errors
        errors = parser.errors
        # Update Tokens
        token_output.delete("1.0", tk.END)
        for token in tokens:
            token_output.insert(tk.END, f"{token}\n")
        # Update AST
        ast_output.delete("1.0", tk.END)
        ast_text = print_ast(ast, indent=0)
        ast_output.insert("1.0", ast_text)
        # Update Parse Table
        parse_table = create_parse_table(grammar, first_sets, follow_sets)
        table_output.delete("1.0", tk.END)
        for non_terminal, rules in parse_table.items():
            rules_str = ""
            for terminal, prods in rules.items():
                for prod in prods:
                    prod_str = ' '.join(prod)
                    rules_str += f"  {terminal} -> {prod_str}\n"
            table_output.insert(tk.END, f"{non_terminal}:\n{rules_str}\n")
        # Update Errors and Semantic Analysis
        if errors:
            error_output.delete("1.0", tk.END)
            for error in errors:
                error_output.insert(tk.END, f"{error}\n")
            semantic_output.delete("1.0", tk.END)  # Clear semantic analysis
        else:
            error_output.delete("1.0", tk.END)
            # Display Symbol Table as Semantic Analysis
            symbol_table_text = print_symbol_table(parser.global_symbol_table, indent=0)
            semantic_output.delete("1.0", tk.END)
            semantic_output.insert("1.0", symbol_table_text)

    def show_parse_table():
        parse_table = create_parse_table(grammar, first_sets, follow_sets)
        table_output.delete("1.0", tk.END)
        for non_terminal, rules in parse_table.items():
            rules_str = ""
            for terminal, prods in rules.items():
                for prod in prods:
                    prod_str = ' '.join(prod)
                    rules_str += f"  {terminal} -> {prod_str}\n"
            table_output.insert(tk.END, f"{non_terminal}:\n{rules_str}\n")

    # Compute FIRST and FOLLOW sets initially
    first_sets = compute_first(grammar)
    follow_sets = compute_follow(grammar, first_sets)

    # GUI
    root = tk.Tk()
    root.title("Compiler Simulation with Semantic Analysis")

    # Input Section
    tk.Label(root, text="Source Code:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
    input_text = scrolledtext.ScrolledText(root, height=10, width=60)
    input_text.grid(row=1, column=0, padx=10, pady=5)

    # Tokenized Output
    tk.Label(root, text="Tokens:").grid(row=0, column=1, sticky=tk.W, padx=10, pady=5)
    token_output = scrolledtext.ScrolledText(root, height=10, width=40)
    token_output.grid(row=1, column=1, padx=10, pady=5)

    # AST Output
    tk.Label(root, text="Abstract Syntax Tree (AST):").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
    ast_output = scrolledtext.ScrolledText(root, height=10, width=60)
    ast_output.grid(row=3, column=0, padx=10, pady=5)

    # Parse Table Output
    tk.Label(root, text="Parse Table:").grid(row=2, column=1, sticky=tk.W, padx=10, pady=5)
    table_output = scrolledtext.ScrolledText(root, height=10, width=40)
    table_output.grid(row=3, column=1, padx=10, pady=5)

    # Errors Output
    tk.Label(root, text="Semantic Errors:").grid(row=4, column=0, sticky=tk.W, padx=10, pady=5)
    error_output = scrolledtext.ScrolledText(root, height=5, width=60, fg="red")
    error_output.grid(row=5, column=0, padx=10, pady=5)

    # Semantic Analysis Output
    tk.Label(root, text="Semantic Analysis (Symbol Table):").grid(row=4, column=1, sticky=tk.W, padx=10, pady=5)
    semantic_output = scrolledtext.ScrolledText(root, height=5, width=40)
    semantic_output.grid(row=5, column=1, padx=10, pady=5)

    # Buttons
    button_frame = tk.Frame(root)
    button_frame.grid(row=6, column=0, columnspan=2, pady=10)

    tk.Button(button_frame, text="Tokenize", command=tokenize_input).grid(row=0, column=0, padx=5)
    tk.Button(button_frame, text="Parse & Analyze", command=parse_input).grid(row=0, column=1, padx=5)
    tk.Button(button_frame, text="Show Parse Table", command=show_parse_table).grid(row=0, column=2, padx=5)

    root.mainloop()


# Example usage of the tokenizer and parser
if __name__ == "__main__":
    # Uncomment the following lines if you want to run the CLI version

    sample_code = open("x.TEAM", "r", encoding="utf-8").read()
    print("Tokenizing...\n")
    tokens = tokenize(sample_code)
    for token in tokens:
        print(token)
    
    print("\nParsing...\n")
    parser = Parser(tokens)
    ast = parser.parse()
    ast_text = print_ast(ast)
    print(ast_text)
    
    print("\nComputing FIRST and FOLLOW sets...\n")
    first = compute_first(grammar)
    follow = compute_follow(grammar, first)
    display_grammar_and_sets(grammar, first, follow)
    
    print("\nSymbol Table:\n")
    symbol_table_text = print_symbol_table(parser.global_symbol_table)
    print(symbol_table_text)
    
    print("Launching GUI Simulation...")
    
    # Run the GUI
    GUI()
