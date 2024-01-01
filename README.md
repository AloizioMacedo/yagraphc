# YAGraphC

![Code Tests](https://github.com/AloizioMacedo/yagraphc/actions/workflows/tests.yml/badge.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/AloizioMacedo/yagraphc/badge.svg?branch=create-ci-coverage)](https://coveralls.io/github/AloizioMacedo/yagraphc?branch=create-ci-coverage)
![Linting](https://github.com/AloizioMacedo/yagraphc/actions/workflows/linting.yml/badge.svg?branch=master)
![Doc Tests](https://github.com/AloizioMacedo/yagraphc/actions/workflows/doctests.yml/badge.svg?branch=master)

Crate for working with Graph data structures and common algorithms on top of it.

The main focus of this crate is **functionality**. Performance is appreciated but
not the main priority. It is intended to fill the gaps in terms of what is not
currently available in the ecosystem. As an example, it is not easy to find
a graph crate which finds a cycle basis of an undirected graph, while this
is trivial in Python's [NetworkX](https://networkx.org/).

## Example

```rust
use yagraphc::prelude::*;
use yagraphc::graph::UnGraph;

let mut graph = UnGraph::default();

graph.add_edge(1, 2, 1);
graph.add_edge(2, 3, 3);
graph.add_edge(3, 4, 2);
graph.add_edge(1, 4, 10);

assert_eq!(graph.dijkstra_with_path(1, 4), Some((vec![1, 2, 3, 4], 6)));
```

## Goals before 1.0.0

The development will keep moving towards the following objectives. Once those
are reached, we will reevaluate how to move forward.

- Main path-finding algorithms implemented
- Main max flow / min cut algorithms implemented
- Full test coverage
