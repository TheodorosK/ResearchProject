import random
import string
import sys
import check_security
from random import randint
from pprint import pprint

sys.setrecursionlimit(500000)

INT_START_RANGE = -999999
INT_END_RANGE = 999999
IDENTIFIER_LENGTH = 5

MAX_DEPTH_EXPRESSION = 2
MAX_DEPTH_COMMAND = 15


RESERVED_KEYWORDS = ['if', 'then', 'else', 'while', 'do']
TAB_SIZE = '    '

# ENABLE_SEED = False
ENABLE_SEED = True

all_vars_asts = []
ass_vars_l_r = []

if ENABLE_SEED:
    SEED = 272306
    random.seed(SEED)
    print('SEED: {}'.format(SEED))


class IntExpr:

    def gen(self):
        """
        Return AST representation for integers.
        The value for the key 'Value' is a randomly generated integer.
        The value range is defined in the constants INT_START_RANGE and
        INT_END_RANGE.
        """
        return {'Kind': 'Int',
                'Value': randint(INT_START_RANGE, INT_END_RANGE)
                }, 1


class VarExpr:

    def gen(self):
        """
        Return AST representation for variables.
        A variable is generated randomly with a random length with a maximum of
        IDENTIFIER_LENGTH, consisting of lowercase and/or uppercase letters.
        """
        global all_vars_asts
        var = ''.join(random.choice(string.ascii_letters)
                      for _ in range(randint(1, IDENTIFIER_LENGTH)))
        while (var.lower() in RESERVED_KEYWORDS):
            var = ''.join(random.choice(string.ascii_letters)
                          for _ in range(randint(1, IDENTIFIER_LENGTH)))

        ast = {'Kind': 'Var',
               'Name': var
               }
        if var not in all_vars_asts:
            all_vars_asts.append(ast)

        return ast, 1


class LiteralExpr:

    def gen(self):
        if randint(0, 1) == 0:
            return IntExpr().gen()
        else:
            return VarExpr().gen()


class NullExpr:

    def gen(self):
        return {'Kind': 'Null',
                'Value': 'Null'
                }, 1


class AddExpr:

    def gen(self, depth):
        depth_left, depth_right = get_rand_depth(depth - 1)

        left, depth_left = ExprGen().gen(depth_left)
        right, depth_right = ExprGen().gen(depth_right)
        depth_ret = max(depth_left, depth_right)

        return {'Kind': 'Add',
                'Left': left,
                'Right': right,
                }, depth_ret + 1


class SubExpr:

    def gen(self, depth):
        depth_left, depth_right = get_rand_depth(depth - 1)

        left, depth_left = ExprGen().gen(depth_left)
        right, depth_right = ExprGen().gen(depth_right)
        depth_ret = max(depth_left, depth_right)

        return {'Kind': 'Sub',
                'Left': left,
                'Right': right,
                }, depth_ret + 1


class EqualExpr:

    def gen(self, depth):
        depth_left, depth_right = get_rand_depth(depth - 1)

        left, depth_left = ExprGen().gen(depth_left)
        right, depth_right = ExprGen().gen(depth_right)
        depth_ret = max(depth_left, depth_right)

        return {'Kind': 'Equal',
                'Left': left,
                'Right': right,
                }, depth_ret + 1


class LessExpr:

    def gen(self, depth):
        depth_left, depth_right = get_rand_depth(depth - 1)

        left, depth_left = ExprGen().gen(depth_left)
        right, depth_right = ExprGen().gen(depth_right)
        depth_ret = max(depth_left, depth_right)

        return {'Kind': 'Less',
                'Left': left,
                'Right': right,
                }, depth_ret + 1


class ExprGen:

    def gen(self, depth):
        if depth == 0:
            return LiteralExpr().gen()

        rnd = randint(0, 3)
        if rnd == 0:
            return AddExpr().gen(depth)
        elif rnd == 1:
            return SubExpr().gen(depth)
        elif rnd == 2:
            return EqualExpr().gen(depth)
        elif rnd == 3:
            return LessExpr().gen(depth)


class ExpressionGenerator:

    def gen(self):
        depth = randint(0, MAX_DEPTH_EXPRESSION)
        # print('Generating expression with depth {}'.format(depth))
        return ExprGen().gen(depth)


class AssignCmd:

    def gen(self, depth):
        global ass_vars_l_r
        left, _ = VarExpr().gen()
        right, _ = ExpressionGenerator().gen()

        # find all vars used on the right side that are not equal to the
        # left side
        right_vars = find_vars(ast=right, vars=[], left_var_name=left['Name'])
        if right_vars is not None:
            # search ass_vars_l_r for left var to figure out if it has already
            # been assigned and try to append new right vars to it's rightvars
            # list
            left_var_found = False
            for i in range(len(ass_vars_l_r)):
                if ass_vars_l_r[i]['Name'] == left['Name']:
                    left_var_found = True
                    for e in right_vars:
                        if e not in ass_vars_l_r[i]['RightVars']:
                            ass_vars_l_r[i]['RightVars'].append(e)

            if not left_var_found and len(right_vars) > 0:
                # left var has not yet been assigned, create a new list entry
                ass_vars_l_r.append({
                    'Name': left['Name'],
                    'RightVars': right_vars
                })

        return {'Kind': 'Assign',
                'Left': left,
                'Right': right,
                }, 0

    def gen_null_assign(self, left):
        null_expr, _ = NullExpr().gen()
        return {'Kind': 'Assign',
                'Left': left,
                'Right': null_expr,
                }, 0


def find_vars(ast, vars, left_var_name):
    """
    Return names of vars from ast for each var that has not the same name as
    the given argument left_var_name, which is the left side of an assignment
    """
    kind = ast.get("Kind")

    if kind == 'Var':
        if ast.get("Name") not in vars and ast.get("Name") != left_var_name:
            vars.append(ast.get("Name"))
        return vars
    elif kind == 'If':
        return find_vars(ast.get("Else"),
                         find_vars(ast.get("Then"),
                                   vars,
                                   left_var_name),
                         left_var_name)
    elif kind == 'While':
        return find_vars(ast.get("Body"),
                         vars,
                         left_var_name)
    elif kind != 'Int':
        return find_vars(ast.get("Left"),
                         find_vars(ast.get("Right"),
                                   vars,
                                   left_var_name),
                         left_var_name)
    else:
        return vars


class SeqCmd:

    def gen(self, depth):
        depth_left, depth_right = get_rand_depth(depth - 1)
        left, depth_left = CmdGen().gen(depth_left)
        right, depth_right = CmdGen().gen(depth_right)
        depth_ret = max(depth_left, depth_right)

        return {'Kind': 'Seq',
                'Left': left,
                'Right': right
                }, depth_ret + 1

    def gen_pre_seq(self, left, right, depth):
        return {'Kind': 'Seq',
                'Left': left,
                'Right': right
                }, depth + 1


class WhileCmd:

    def gen(self, depth):
        cond, _ = ExpressionGenerator().gen()
        body, depth_body = CmdGen().gen(depth - 1)

        return {'Kind': 'While',
                'Condition': cond,
                'Body': body,
                }, depth_body + 1


class IfCmd:

    def gen(self, depth):
        cond, _ = ExpressionGenerator().gen()

        depth_then, depth_else = get_rand_depth(depth - 1)

        then, depth_then = CmdGen().gen(depth_then)
        _else, depth_else = CmdGen().gen(depth_else)
        depth_ret = max(depth_then, depth_else)

        return {'Kind': 'If',
                'Condition': cond,
                'Then': then,
                'Else': _else,
                }, depth_ret + 1


class CmdGen:

    def gen(self, depth):
        if depth == 0:
            return AssignCmd().gen(depth)

        rnd = randint(0, 2)
        if rnd == 0:
            return WhileCmd().gen(depth)
        elif rnd == 1:
            return IfCmd().gen(depth)
        elif rnd == 2:
            return SeqCmd().gen(depth)


class CommandGenerator:

    def gen(self, gen_secure):
        global all_vars_asts
        all_vars_asts = []
        depth = randint(1, MAX_DEPTH_COMMAND)
        ast, depth = CmdGen().gen(depth)

        identifier_storage = []
        if not gen_secure:
            # if there are assignments which use vars on the right side
            if len(ass_vars_l_r) > 0:
                # choose one or a random amount of assignments which should be
                # marked as insecure (left var L, at least one right var H)
                no_of_insec_ass = 1
                # no_of_insec_ass = randint(1, len(ass_vars_l_r))
                assigns = random.sample(ass_vars_l_r, no_of_insec_ass)
                for e in assigns:
                    # choose one or a random amount of vars from the right side
                    # and set each of them as H
                    no_of_r_vars = 1
                    # no_of_r_vars = randint(1, len(e['RightVars']))
                    r_vars = random.sample(e['RightVars'], no_of_r_vars)
                    for ee in r_vars:
                        # mark var from right as H, e. g.  ( ... := ... x ...)
                        identifier_storage.append({
                            "Identifier": ee,
                            "Security": "H"
                        })
                    # mark var from left as L, e. g.  ( x := ...)
                    identifier_storage.append({
                        "Identifier": e['Name'],
                        "Security": "L"
                    })
        for e in all_vars_asts:
            if not in_identifier_storage(e['Name'], identifier_storage):
                # if var is not in identifier_storage, add it
                # with a random security class
                identifier_storage.append({
                    "Identifier": e['Name'],
                    "Security": random.choice(['H', 'L'])
                })

            # preassign var with NullExpr
            left, depth_left = AssignCmd().gen_null_assign(e)
            ast, depth = SeqCmd().gen_pre_seq(left, ast, depth_left + depth)

        sec_valid, _ = check_security.security(ast, identifier_storage)

        if sec_valid != gen_secure:
            return CommandGenerator().gen(gen_secure=gen_secure)
        else:
            print('Generated command with depth {}'.format(depth))
            return ast, depth, identifier_storage, sec_valid


def in_identifier_storage(var, identifier_storage):
    for e in identifier_storage:
        if var == e['Identifier']:
            return True
    return False


def get_rand_depth(depth):
    if randint(0, 1) == 0:
        return depth, randint(0, depth)
    else:
        return randint(0, depth), depth


def prettyprint_singleline(ast):
    """Return AST as human readable single-line string with bracketing"""
    code = ''
    if 'Kind' in ast:
        if ast['Kind'] == 'If':
            code += "if {} then {{{}}} else {{{}}}".format(
                prettyprint_singleline(ast['Condition']),
                prettyprint_singleline(ast['Then']),
                prettyprint_singleline(ast['Else']))
        elif ast['Kind'] == 'While':
            code += "while {} do {{{}}}".format(
                prettyprint_singleline(ast['Condition']),
                prettyprint_singleline(ast['Body']))
        elif ast['Kind'] == 'Int':
            code += str(ast['Value'])
        elif ast['Kind'] == 'Var':
            code += str(ast['Name'])
        elif ast['Kind'] == 'Null':
            code += str(ast['Value'])
        else:
            code += "({} {} {})".format(
                prettyprint_singleline(ast['Left']),
                get_center(ast['Kind']),
                prettyprint_singleline(ast['Right']))
    return code


def prettyprint_multiline_indented(ast, level=0):
    """
    Return AST as human readable multi-line string with bracketing and
    indentation
    """
    code = ''
    if 'Kind' in ast:
        if ast['Kind'] == 'If':
            code += "{}if {} then {{\n{}\n{}}} else {{\n{}\n{}}}".format(
                get_tabs(level),
                prettyprint_multiline_indented(ast['Condition'], level),
                prettyprint_multiline_indented(ast['Then'], level + 1),
                get_tabs(level),
                prettyprint_multiline_indented(ast['Else'], level + 1),
                get_tabs(level))
        elif ast['Kind'] == 'While':
            code += "{}while {} do {{\n{}\n{}}}".format(
                get_tabs(level),
                prettyprint_multiline_indented(ast['Condition'], level),
                prettyprint_multiline_indented(ast['Body'], level + 1),
                get_tabs(level))
        elif ast['Kind'] == 'Assign':
            code += "{}{} {} {}".format(
                get_tabs(level),
                prettyprint_multiline_indented(ast['Left'], level),
                get_center(ast['Kind']),
                prettyprint_multiline_indented(ast['Right'], level))
        elif ast['Kind'] == 'Seq':
            code += "{}{}\n{}".format(
                prettyprint_multiline_indented(ast['Left'], level),
                get_center(ast['Kind']),
                prettyprint_multiline_indented(ast['Right'], level))
        elif ast['Kind'] == 'Add':
            code += "({} {} {})".format(
                prettyprint_multiline_indented(ast['Left'], level),
                get_center(ast['Kind']),
                prettyprint_multiline_indented(ast['Right'], level))
        elif ast['Kind'] == 'Sub':
            code += "({} {} {})".format(
                prettyprint_multiline_indented(ast['Left'], level),
                get_center(ast['Kind']),
                prettyprint_multiline_indented(ast['Right'], level))
        elif ast['Kind'] == 'Less':
            code += "({} {} {})".format(
                prettyprint_multiline_indented(ast['Left'], level),
                get_center(ast['Kind']),
                prettyprint_multiline_indented(ast['Right'], level))
        elif ast['Kind'] == 'Equal':
            code += "({} {} {})".format(
                prettyprint_multiline_indented(ast['Left'], level),
                get_center(ast['Kind']),
                prettyprint_multiline_indented(ast['Right'], level))
        elif ast['Kind'] == 'Int':
            code += str(ast['Value'])
        elif ast['Kind'] == 'Var':
            code += str(ast['Name'])
        elif ast['Kind'] == 'Null':
            code += str(ast['Value'])
        else:
            raise RuntimeError("Unknown kind {}".format(ast['Kind']))
    return code


def get_tabs(level):
    """Return tab string for the given level"""
    tabs = ''
    for i in range(level):
        tabs += TAB_SIZE
    return tabs


def get_center(kind):
    """Return the symbol(s) associated with each kind"""
    if kind == 'Assign':
        return ':='
    elif kind == 'Seq':
        return '; '
    elif kind == 'Add':
        return '+'
    elif kind == 'Sub':
        return '-'
    elif kind == 'Equal':
        return '=='
    elif kind == 'Less':
        return '<'
    else:
        raise RuntimeError("Unknown kind {}".format(kind))


def main():
    gen_secure_code = True
    ast, depth, identifier_storage, secure = CommandGenerator().gen(gen_secure_code)
    print('\nIdentifier together with their security classes')
    pprint(identifier_storage)
    print('\n')
    print(prettyprint_multiline_indented(ast))
    if secure:
        print('\nprogram is secure')
    else:
        print('\nprogram is insecure')


if __name__ == "__main__":
    main()
