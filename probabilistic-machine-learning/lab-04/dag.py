from __future__ import annotations

class DAGNode:
    def __init__(self, name: str, children: set[DAGNode] = set(), parents: set[DAGNode] = set()):
        self.name = name
        self.children = children
        self.parents = parents
    
    def __repr__(self):
        return '<' + self.name + '>'
    
    def descendants(self) -> set[DAGNode]:
        if len(self.children) == 0:
            return set()
        result = set()
        for child in self.children:
            result.add(child)
            result |= child.descendants()
        return result
    
    # return self.children | set(child for child in self.descendants())

class DAG:
    def __init__(self, V: dict[str, DAGNode], roots: list[str]):
        self.V = V
        self.roots = roots

def parse_graph(input: str) -> DAG:
    terms = [x.strip() for x in input.split(';')]
    vertices = {}
    non_root_nodes = set()
    
    for term in terms:
        atoms = [x.strip() for x in term.split('->')]
        parents = set()
        children = set()
        
        for atom in atoms:
            nodes = [x.strip() for x in atom[1:-1].split(',')] if atom.startswith('(') else [atom]
            
            for node in nodes:
                if node not in vertices:
                    vertices[node] = DAGNode(node)
                children.add(vertices[node])
                vertices[node].parents = vertices[node].parents | parents.copy()
            
            if parents:
                for parent in parents:
                    parent.children = parent.children | children.copy()
                non_root_nodes |= set(c.name for c in children)
            
            parents = children.copy()
            children = set()
    
    return DAG(vertices, list(set(vertices.keys()).difference(non_root_nodes)))