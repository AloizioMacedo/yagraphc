//! # Introduction
//!
//! Crate for working with Graph data structures and common algorithms on top of it.
//!
//! The main focus of this crate is **functionality**. Performance is appreciated but
//! not the main priority. It is intended to fill the gaps in terms of what is not
//! currently available in the ecosystem. As an example, it is not easy to find
//! a graph crate which finds a cycle basis of an undirected graph, while this
//! is trivial in Python's [networkx](https://networkx.org/).
//!
//! ## Example
//!
//! ```rust
//! use yagraphc::prelude::*;
//! use yagraphc::graph::UnGraph;
//!
//! let mut graph = UnGraph::default();
//!
//! graph.add_edge(1, 2, 1);
//! graph.add_edge(2, 3, 3);
//! graph.add_edge(3, 4, 2);
//! graph.add_edge(1, 4, 10);
//!
//! assert_eq!(graph.dijkstra_with_path(1, 4), Some((vec![1, 2, 3, 4], 6)));
//! ```

pub mod graph;

pub mod prelude {
    pub use super::graph::traits::ArithmeticallyWeightedGraph;
    pub use super::graph::traits::GraphBuilding;
    pub use super::graph::traits::Traversable;
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_prelude() {
        use crate::graph::UnGraph;
        use crate::prelude::*;

        let mut graph = UnGraph::default();
        graph.add_edge(1, 2, ());
    }
}
