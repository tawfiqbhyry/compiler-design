import re
import tkinter as tk
from tkinter import scrolledtext

# Define token specifications
TOKEN_SPECIFICATION = [
    ('COMMENT', r'//[^\n]*'),                                 # Single-line comments
    ('KEYWORD', r'\b(if|else|while|for|return|function|then|do|break|continue|var|let|const|switch|case|import|export|from|to)\b'),
    ('BOOLEAN', r'\b(true|false)\b'),                         # Boolean values
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
    ('SEMICOLON', r';'),                                      # Semicolon
    ('COLON', r':'),                                          # Colon
    ('COMMA', r','),                                          # Comma
    ('WHITESPACE', r'[ \t]+'),                                # Spaces and tabs
    ('NEWLINE', r'\n'),                                       # Line endings
]

grammar = {
    "Program": [["StatementList"]],
    "StatementList": [["Statement", "StatementList"], ["ε"]],
    "Statement": [["VariableDeclaration"], ["FunctionDeclaration"], ["IfStatement"], ["WhileStatement"]],
    "VariableDeclaration": [["var", "IDENTIFIER", "ASSIGN", "Expression", "SEMICOLON"],
                            ["let", "IDENTIFIER", "ASSIGN", "Expression", "SEMICOLON"],
                            ["const", "IDENTIFIER", "ASSIGN", "Expression", "SEMICOLON"]],
    "FunctionDeclaration": [["function", "IDENTIFIER", "LPAREN", "ParameterList", "RPAREN", "Block"]],
    "IfStatement": [["if", "LPAREN", "Expression", "RPAREN", "Block", "else", "Block"]],
    "WhileStatement": [["while", "LPAREN", "Expression", "RPAREN", "Block"]],
    "Expression": [["Term", "OPERATOR", "Expression"], ["Term"]],
    "Term": [["IDENTIFIER"], ["Literal"]],
    "ParameterList": [["IDENTIFIER", "COMMA", "ParameterList"], ["IDENTIFIER"], ["ε"]],
    "Block": [["LBRACE", "StatementList", "RBRACE"]],
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
                            first_of_sequence(next_symbols, first)
                        )

                        # If next_symbols derive ε, add FOLLOW of current non-terminal
                        if "ε" in first_of_sequence(next_symbols, first):
                            follow[symbol].update(follow[non_terminal])

                        if len(follow[symbol]) > before:
                            changed = True
    return follow

def first_of_sequence(sequence, first):
    result = set()
    for symbol in sequence:
        result.update(first.get(symbol, {symbol}))
        if "ε" not in first.get(symbol, {symbol}):
            break
    else:
        result.add("ε")
    return result


def display_grammar_and_sets(grammar, first, follow):
    print("Grammar (BNF):")
    for non_terminal, productions in grammar.items():
        print(f"{non_terminal} ::= {' | '.join(' '.join(p) for p in productions)}")

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
    def __init__(self, kind, name, initializer):
        self.kind = kind  # 'var', 'let', or 'const'
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
        if self.current_token and self.current_token.type == 'IDENTIFIER':
            var_name = self.current_token.value
            declared_at = (self.current_token.line, self.current_token.column)
            # Attempt to define the variable in the current symbol table
            if not self.symbol_table.define(var_name, 'variable', declared_at=declared_at):
                self.error(f"Variable '{var_name}' is already declared in scope '{self.symbol_table.scope_name}'.")
            self.consume('IDENTIFIER')
            initializer = None
            if self.current_token and self.current_token.type == 'ASSIGN':
                self.consume('ASSIGN')
                initializer = self.expression()
            if not self.consume('SEMICOLON'):
                return None
            return VariableDeclaration(kind, var_name, initializer)
        else:
            self.error("Expected identifier in variable declaration.")
            return None

    def function_declaration(self):
        self.consume('KEYWORD')  # 'function'
        if self.current_token and self.current_token.type == 'IDENTIFIER':
            func_name = self.current_token.value
            declared_at = (self.current_token.line, self.current_token.column)
            # Attempt to define the function in the current symbol table
            if not self.symbol_table.define(func_name, 'function', declared_at=declared_at):
                self.error(f"Function '{func_name}' is already declared in scope '{self.symbol_table.scope_name}'.")
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
                if not self.symbol_table.define(param_name, 'parameter', declared_at=param_declared_at):
                    self.error(f"Parameter '{param_name}' is already declared in scope '{self.symbol_table.scope_name}'.")
            # Parse the function body
            body = self.block(scope_name=func_scope_name)
            # Restore the previous symbol table
            self.symbol_table = self.symbol_table.parent
            return FunctionDeclaration(func_name, param_names, body)
        else:
            self.error("Expected function name.")
            return None

    def parameter_list(self):
        params = []
        while self.current_token and self.current_token.type == 'IDENTIFIER':
            param_name = self.current_token.value
            param_token = self.current_token
            params.append((param_name, param_token))
            self.consume('IDENTIFIER')
            if self.current_token and self.current_token.type == 'COMMA':
                self.consume('COMMA')
            else:
                break
        return params

    def block(self, scope_name="block"):
        if not self.consume('LBRACE'):
            return None
        # Create a new symbol table for the block
        self.symbol_table = SymbolTable(parent=self.symbol_table, scope_name=scope_name)
        statements = self.statement_list()
        if not self.consume('RBRACE'):
            return None
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
            self.consume('RPAREN')
            return expr
        else:
            self.error("Unexpected token in expression.")
            self.consume(token.type)  # Attempt to recover
            return None

    def function_call(self):
        func_name = self.current_token.value
        self.consume('IDENTIFIER')
        self.consume('LPAREN')
        args = self.argument_list()
        self.consume('RPAREN')
        # Check if the function is defined
        symbol = self.symbol_table.lookup(func_name)
        if not symbol:
            self.error(f"Call to undeclared function '{func_name}'.")
        elif symbol.kind != 'function':
            self.error(f"Identifier '{func_name}' is not a function.")
        return FunctionCall(func_name, args)

    def argument_list(self):
        args = []
        if self.current_token and self.current_token.type != 'RPAREN':
            args.append(self.expression())
            while self.current_token and self.current_token.type == 'COMMA':
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
    result = ""

    if isinstance(node, Program):
        result += f"{prefix}Program\n"
        for stmt in node.statements:
            result += print_ast(stmt, indent + 1)
        return result
    elif isinstance(node, VariableDeclaration):
        result += f"{prefix}VariableDeclaration ({node.kind}): {node.name}\n"
        if node.initializer:
            result += print_ast(node.initializer, indent + 1)
        return result
    elif isinstance(node, FunctionDeclaration):
        result += f"{prefix}FunctionDeclaration: {node.name} Params: {', '.join(node.params)}\n"
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
        result += f"{prefix}Assignment: {node.identifier} =\n"
        result += print_ast(node.value, indent + 1)
        return result
    elif isinstance(node, FunctionCall):
        result += f"{prefix}FunctionCall: {node.name}\n"
        for arg in node.arguments:
            result += print_ast(arg, indent + 1)
        return result
    else:
        result += f"{prefix}Unknown node type: {type(node).__name__}\n"
        return result

# Symbol Table Printer for visualization
def print_symbol_table(symbol_table, indent=0):
    prefix = '  ' * indent
    result = f"{prefix}Scope: {symbol_table.scope_name}\n"
    for symbol in symbol_table.symbols.values():
        result += f"{prefix}  {symbol.name} : {symbol.kind}, Declared at: Line {symbol.declared_at[0]}, Column {symbol.declared_at[1]}\n"
    for child in symbol_table.children:
        result += print_symbol_table(child, indent + 1)
    return result

def create_parse_table(grammar, first, follow):
    parse_table = {}
    for non_terminal in grammar:
        parse_table[non_terminal] = {}
        for production in grammar[non_terminal]:
            for terminal in first_of_sequence(production, first):
                if terminal != "ε":
                    parse_table[non_terminal][terminal] = production
            if "ε" in first_of_sequence(production, first):
                for terminal in follow[non_terminal]:
                    parse_table[non_terminal][terminal] = production
    return parse_table

def display_parse_table(parse_table):
    print("Parse Table:")
    for non_terminal, row in parse_table.items():
        print(f"{non_terminal}: {row}")

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
            table_output.insert(tk.END, f"{non_terminal}: {rules}\n")
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
            table_output.insert(tk.END, f"{non_terminal}: {rules}\n")

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
    
    print("GUI Simulation: ")
    
    # Run the GUI
    GUI()
