""" This module provides the tools necessary to load filter definitions from
    plaintext files. For a description of the definition language, refer to
    README.md.
"""

__all__ = ["Tokenizer", "Parser", "Instantiator", "LoadFilter"]

from string import ascii_letters, digits
import filter

ALPHANUM = ascii_letters + digits
NUMERIC  = digits + "."

class Tokenizer (object):
    """ Splits a string into a list of tokens accepted by the language. The
        tokens are basically just literals (strings and floats) and symbols
        (parens, brackets, braces, comma, equal sign).
    """

    def tokenize(self, string):
        """ Returns the tokenized string
        """
    
        tokens  = []
        current = (None, [])
        
        for c in string:
        
            if c in NUMERIC and (current[0] != "identifier"):
                t = "numeric"
            elif c in ALPHANUM and (current[0] != "numeric"):
                t = "identifier"
            elif c in "[](){},=":
                t = c
            else:
                t = None
            
            if current[0] != t:
                if current[0] != None:
                    tokens.append((current[0], "".join(current[1])))
                current = (t, [])
            current[1].append(c)
        
        if current[0] != None:
            tokens.append((current[0], "".join(current[1])))
        
        return tokens
                    

class Parser (object):
    """ Accepts tokens and parses them into a tree structure composed of Python
        objects (lists, dicts, and primitives).
    """

    class Chain (list):
        """ Represents a FilterChain
        """
        pass
    
    class Stack (list):
        """ Represents a FilterStack
        """
        pass
    
    class Object (dict):
        """ Represents a configuration object for a filter
        """
        pass
    
    class Filter (object):
        """ Represents a basic filter object, optionally with configuration
        """
        def __init__(self, name):
            self.name = name
            self.args = None
            
    def parseContainer(self, t, root):
        """ Parses a container type, i.e. a Chain or a Stack. The parameter 't'
            must be the corresponding type. The param 'root' is the root of the
            pre-processed tree.
        """
        
        c = t()
        
        funcs = {
            "chain": self.parseChain,
            "stack": self.parseStack,
            "object": self.parseObject
        }
        
        pending = None
        
        for node in root:
            if type(node) == tuple:
                if node[0] in funcs:
                    if node[0] == "object":
                        if pending is None or type(pending) != Parser.Filter:
                            raise RuntimeError("unexpected config object in container")
                        else:
                            pending.args = self.parseObject(node[2])
                    else:
                        if pending is None:
                            pending = funcs[node[0]](node[2])
                        else:
                            raise RuntimeError("missing expected ',' in container")
                else:
                    raise RuntimeError("unexpected %s found in container" % node[0])
            elif node  == ",":
                if pending is None:
                    raise RuntimeError("found unexpected ',' in container")
                else:
                    c.append(pending)
                    pending = None
            elif type(node) == str:
                pending = self.parseFilter(node)
            else:
                print("found unexpected item in container: %s" % str(node))
        
        if pending is not None:
            c.append(pending)
        
        return c
    
    def parseChain(self, root):
        """ Parses a Chain object using parseContainer
        """
        return self.parseContainer(Parser.Chain, root)
    
    def parseStack(self, root):
        """ Parses a Stack object using parseContainer
        """
        return self.parseContainer(Parser.Stack, root)
    
    def parseLiteral(self, root):
        """ Parses a literal -- if it can be converted to a float, it will be;
            otherwise it will be kept as a string
        """
        res = None
        try:
            res = float(root)
        except ValueError:
            res = root
        return res
    
    def parseList(self, root):
        """ Parses a list of literals
        """
        c = []
        
        pending = None
        
        for node in root:
            if node == ",":
                if pending is None:
                    raise RuntimeError("found unexpected ',' in container")
                else:
                    c.append(pending)
                    pending = None
            elif type(node) == str:
                if pending is None:
                    pending = self.parseLiteral(node)
                else:
                    raise RuntimeError("missing expected ',' in container")
            else:
                print("found unexpected item in container: %s" % str(node))
        
        if pending is not None:
            c.append(pending)
        
        return c
    
    def parseObject(self, root):
        """ Parses a configuration object
        """
        p = Parser.Object()
        
        key = None
        equ = False
        val = None
        
        for node in root:
            if type(node) == tuple:
                if node[0] == "stack":
                    if equ is False or val is not None:
                        raise RuntimeError("illegal placement of list value inside config object")
                    else:
                        val = self.parseList(node[2])
                else:
                    raise RuntimeError("found illegal container inside config object")
            elif node == "=":
                if key is None or equ is True or val is not None:
                    raise RuntimeError("illegal placement of '=' inside config object")
                else:
                    equ = True
            elif node == ",":
                if val is None:
                    raise RuntimeError("illegal placement of ',' inside config object")
                p[key] = val
                key, equ, val = None, False, None
            elif type(node) == str:
                if val is None:
                    if key is None:
                        key = node
                    elif equ is True and val is None:
                        val = self.parseLiteral(node)
                    else:
                        raise RuntimeError("illegal placement of key inside config object")
                else:
                    raise RuntimeError("illegal placement of key inside config object")
            else:
                raise RuntimeError("found unknown item %s of %s" % (str(node), type(node)))
                
        if val is not None:
            p[key] = val
        elif key is not None:
            raise RuntimeError("unfinished key in config object")
                
        return p
    
    def parseFilter(self, name):
        """ Parses a Filter object
        """
        return Parser.Filter(name)
    
    def parse(self, tokens):
        """ Parses a token stream. First, the stream is converted into an
            intermediate representation of nested lists of tuples. The other
            parse* methods above handle the conversion of this intermediate
            representation into the actual parsed tree.
        """
        
        root = (None, None, [])
        cur = root

        parens = {
            "(" : ("chain",  True),
            ")" : ("chain",  False),
            "[" : ("stack",  True),
            "]" : ("stack",  False),
            "{" : ("object", True),
            "}" : ("object", False)
        }
        
        funcs = {
            "chain": self.parseChain,
            "stack": self.parseStack,
            "identifier": self.parseFilter
        }
        
        for t, content in tokens:
            
            if t in parens:
                ptype, mode = parens[t]
                if mode:
                    next = (ptype, cur, [])
                    cur[2].append(next)
                    cur = next
                elif ptype == cur[0]:
                    cur = cur[1]
                else:
                    raise RuntimeError("mismatched brackets - %s and %s" % (ptype, cur[0]))
            else:
                cur[2].append(content)
        
        if cur is not root:
            raise RuntimeError("unclosed %s bracket somewhere" % (cur[0]))
        
        return self.parseChain(root[2])

class Instantiator (object):
    """ Instantiates an actual filter object based on the contents of a parse
        tree.
    """

    def instantiate(self, root):
        """ Recursively converts the parse tree (root) into an actual object
            from ppg.filter.
        """
    
        if isinstance(root, Parser.Chain):
            for i,v in enumerate(root):
                root[i] = self.instantiate(v)
            return filter.FilterChain(*root)
            
        elif isinstance(root, Parser.Stack):
            for i,v in enumerate(root):
                root[i] = self.instantiate(v)
            return filter.FilterStack(*root)
            
        elif isinstance(root, Parser.Object):
            for k, v in root.items():
                root[k] = self.instantiate(v)
            return root
        
        elif isinstance(root, Parser.Filter):
            if not root.name in filter.__all__:
                raise RuntimeError("unknown filter %s" % root.name)
            kwargs = root.args or {}
            return filter.__getattribute__(root.name)(**kwargs)
        
        else:
            return root

def LoadFilter(filterString):
    """ Convenience method for converting a string directly into a real, usable
        Filter object.
    """
    return Instantiator().instantiate(Parser().parse(Tokenizer().tokenize(filterString)))
