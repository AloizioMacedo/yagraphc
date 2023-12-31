//! Traits that correspond to the main functionalities of the crate.
//!
//! `Graph` is the main trait for working with general graph traversal, such as
//! BFS and DFS.
//!
//! `ArithmeticallyWeightedGraph` is the main trait for working with path finding,
//! such as Dijkstra's algorithm or A*. It is also intended to handle more general
//! algorithms that rely on arithmetical weights in the future.
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::hash::Hash;

use thiserror::Error;

#[derive(Error, Debug)]
#[error("node not found")]
pub struct NodeNotFound;

pub struct BfsIter<'a, T, W> {
    pub(crate) queue: VecDeque<(T, usize)>,
    pub(crate) visited: HashSet<T>,
    pub(crate) graph: &'a dyn Traversable<T, W>,
}

impl<'a, T, W> Iterator for BfsIter<'a, T, W>
where
    T: Clone + Copy + Hash + PartialEq + Eq,
    W: Clone + Copy,
{
    type Item = (T, usize);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, depth)) = self.queue.pop_front() {
            if self.visited.contains(&node) {
                continue;
            }

            self.visited.insert(node);

            for (next, _) in self.graph.edges(&node) {
                if self.visited.contains(&next) {
                    continue;
                } else {
                    self.queue.push_back((next, depth + 1))
                }
            }

            return Some((node, depth));
        }
        None
    }
}

pub struct DfsIter<'a, T, W> {
    pub(crate) queue: VecDeque<(T, usize)>,
    pub(crate) visited: HashSet<T>,
    pub(crate) graph: &'a dyn Traversable<T, W>,
}

impl<'a, T, W> Iterator for DfsIter<'a, T, W>
where
    T: Clone + Copy + Hash + PartialEq + Eq,
    W: Clone + Copy,
{
    type Item = (T, usize);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, depth)) = self.queue.pop_front() {
            if self.visited.contains(&node) {
                continue;
            }

            self.visited.insert(node);

            for (next, _) in self.graph.edges(&node) {
                if self.visited.contains(&next) {
                    continue;
                } else {
                    self.queue.push_front((next, depth + 1))
                }
            }

            return Some((node, depth));
        }

        None
    }
}

pub struct PostOrderDfsIter<T> {
    pub(crate) queue: VecDeque<(T, usize)>,
}

impl<'a, T> PostOrderDfsIter<T>
where
    T: Clone + Copy + Hash + PartialEq + Eq,
{
    pub fn new<W>(graph: &'a dyn Traversable<T, W>, from: T) -> Self
    where
        W: Clone + Copy,
    {
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();

        dfs_post_order(graph, from, 0, &mut queue, &mut visited);

        Self { queue }
    }
}

fn dfs_post_order<T, W>(
    graph: &dyn Traversable<T, W>,
    node: T,
    depth: usize,
    queue: &mut VecDeque<(T, usize)>,
    visited: &mut HashSet<T>,
) where
    T: Clone + Copy + Hash + PartialEq + Eq,
    W: Clone + Copy,
{
    visited.insert(node);
    for (next, _) in graph.edges(&node) {
        if visited.contains(&next) {
            continue;
        }

        dfs_post_order(graph, next, depth + 1, queue, visited);
    }

    queue.push_back((node, depth));
}

impl<T> Iterator for PostOrderDfsIter<T>
where
    T: Clone + Copy + Hash + PartialEq + Eq,
{
    type Item = (T, usize);

    fn next(&mut self) -> Option<Self::Item> {
        self.queue.pop_front()
    }
}

pub struct NodeIter<'a, T> {
    pub(crate) nodes_iter: std::collections::hash_set::Iter<'a, T>,
}

impl<T> Iterator for NodeIter<'_, T>
where
    T: Clone + Copy,
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.nodes_iter.next().copied()
    }
}

pub enum EdgeIterType<'a, T, W> {
    EdgeIter(EdgeIter<'a, T, W>),
    EdgeIterVec(EdgeIterVec<'a, T, W>),
}

impl<'a, T, W> Iterator for EdgeIterType<'a, T, W>
where
    T: Clone + Copy,
    W: Clone + Copy,
{
    type Item = (T, W);
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            EdgeIterType::EdgeIter(iter) => iter.next(),
            EdgeIterType::EdgeIterVec(iter) => iter.next(),
        }
    }
}

pub struct EdgeIter<'a, T, W> {
    pub(crate) edge_iter: std::collections::hash_map::Iter<'a, T, W>,
}

impl<T, W> Iterator for EdgeIter<'_, T, W>
where
    T: Clone + Copy,
    W: Clone + Copy,
{
    type Item = (T, W);
    fn next(&mut self) -> Option<Self::Item> {
        self.edge_iter.next().map(copy_tuple)
    }
}

pub struct EdgeIterVec<'a, T, W> {
    pub(crate) edge_iter: core::slice::Iter<'a, (T, W)>,
}

impl<T, W> Iterator for EdgeIterVec<'_, T, W>
where
    T: Clone + Copy,
    W: Clone + Copy,
{
    type Item = (T, W);
    fn next(&mut self) -> Option<Self::Item> {
        self.edge_iter.next().copied()
    }
}

pub trait GraphBuilding<T, W>
where
    T: Clone + Copy + Eq + Hash + PartialEq,
    W: Clone + Copy,
{
    fn add_edge(&mut self, from: T, to: T, weight: W);

    fn add_node(&mut self, node: T) -> bool;

    fn remove_edge(&mut self, from: T, to: T) -> Result<(), NodeNotFound>;

    fn remove_node(&mut self, node: T) -> Result<(), NodeNotFound>;

    fn has_edge(&self, from: T, to: T) -> bool;
}

pub trait Traversable<T, W>
where
    T: Clone + Copy + Eq + Hash + PartialEq,
    W: Clone + Copy,
{
    /// Iterates over edges of the node as the target nodes and the edge weight.
    ///
    /// If the graph is undirected, should return the nodes that are connected to it by
    /// an edge.
    ///
    /// If the graph is directed, should return the outbound nodes.
    ///
    /// # Examples
    /// ```rust
    /// use yagraphc::graph::{UnGraph, DiGraph};
    /// use yagraphc::graph::traits::{GraphBuilding, Traversable};
    ///
    /// let mut graph = UnGraph::default();
    ///
    /// graph.add_edge(1, 2, ());
    /// graph.add_edge(2, 3, ());
    ///
    /// let edges = graph.edges(&2);
    ///
    /// assert_eq!(edges.count(), 2);
    ///
    /// let mut graph = DiGraph::default();
    ///
    /// graph.add_edge(1, 2, ());
    /// graph.add_edge(2, 3, ());
    ///
    /// let edges = graph.edges(&2);
    ///
    /// assert_eq!(edges.count(), 1);
    fn edges(&self, n: &T) -> EdgeIterType<T, W>;

    fn edge_weight(&self, from: T, to: T) -> Result<W, NodeNotFound>;

    /// Iterates over inbound-edges of the node as the target nodes and the edge weight.
    ///
    /// If the graph is undirected, should return the nodes that are connected to it by
    /// an edge. Thus, it is equivalent to `edges` in that case.
    ///
    /// If the graph is directed, should return the inbound nodes.
    ///
    /// # Examples
    /// ```rust
    /// use yagraphc::graph::{UnGraph, DiGraph};
    /// use yagraphc::graph::traits::{GraphBuilding, Traversable};
    ///
    /// let mut graph = UnGraph::default();
    ///
    /// graph.add_edge(1, 2, ());
    /// graph.add_edge(2, 3, ());
    /// graph.add_edge(4, 2, ());
    ///
    /// let edges = graph.in_edges(&2);
    ///
    /// assert_eq!(edges.count(), 3);
    ///
    /// let mut graph = DiGraph::default();
    ///
    /// graph.add_edge(1, 2, ());
    /// graph.add_edge(2, 3, ());
    /// graph.add_edge(4, 2, ());
    ///
    /// let edges = graph.in_edges(&2);
    ///
    /// assert_eq!(edges.count(), 2);
    fn in_edges(&self, n: &T) -> EdgeIterType<T, W>;

    /// Returns an iterator over all nodes.
    ///
    /// # Examples
    /// ```rust
    /// use yagraphc::graph::UnGraph;
    /// use yagraphc::graph::traits::{GraphBuilding, Traversable};
    ///
    /// let mut graph = UnGraph::default();
    ///
    /// graph.add_edge(1, 2, ());
    /// graph.add_edge(2, 3, ());
    ///
    /// graph.add_node(4);
    ///
    /// assert_eq!(graph.nodes().count(), 4);
    ///
    /// graph.add_node(2);
    /// assert_eq!(graph.nodes().count(), 4);
    fn nodes(&self) -> NodeIter<T>;

    /// Returns an iterator of nodes in breadth-first order.
    ///
    /// Iterator includes the depth at which the nodes were found. Nodes at the
    /// same depth might be randomly shuffled.
    ///
    /// # Examples
    /// ```
    /// use yagraphc::graph::UnGraph;
    /// use yagraphc::graph::traits::{GraphBuilding, Traversable};
    ///
    /// let mut graph = UnGraph::new();
    ///
    /// graph.add_edge(1, 2, ());
    /// graph.add_edge(1, 3, ());
    /// graph.add_edge(2, 4, ());
    /// graph.add_edge(2, 5, ());
    ///
    /// let bfs = graph.bfs(1);
    ///
    /// let depths = bfs.map(|(_, depth)| depth).collect::<Vec<_>>();
    ///
    /// assert_eq!(depths, vec![0, 1, 1, 2, 2]);
    fn bfs(&self, from: T) -> BfsIter<T, W>
    where
        Self: Sized,
    {
        let visited = HashSet::new();

        let mut queue = VecDeque::new();
        queue.push_front((from, 0));

        BfsIter {
            queue,
            visited,
            graph: self,
        }
    }

    /// Returns an iterator of nodes in depth-first order, in pre-order.
    ///
    /// Iterator includes the depth at which the nodes were found. Order is not
    /// deterministic.
    ///
    /// # Examples
    /// ```
    /// use yagraphc::graph::UnGraph;
    /// use yagraphc::graph::traits::{GraphBuilding, Traversable};
    ///
    /// let mut graph = UnGraph::new();
    ///
    /// graph.add_edge(1, 2, ());
    /// graph.add_edge(2, 3, ());
    /// graph.add_edge(3, 4, ());
    /// graph.add_edge(1, 5, ());
    /// graph.add_edge(5, 6, ());
    ///
    /// let dfs = graph.dfs(1);
    ///
    /// let depths = dfs.map(|(node, _)| node).collect::<Vec<_>>();
    ///
    /// assert!(matches!(depths[..], [1, 2, 3, 4, 5, 6] | [1, 5, 6, 2, 3, 4]));
    fn dfs(&self, from: T) -> DfsIter<T, W>
    where
        Self: Sized,
    {
        let visited = HashSet::new();

        let mut queue = VecDeque::new();
        queue.push_front((from, 0));

        DfsIter {
            queue,
            visited,
            graph: self,
        }
    }

    /// Returns an iterator of nodes in depth-first order, in post-order.
    ///
    /// Iterator includes the depth at which the nodes were found. Order is not
    /// deterministic.
    ///
    /// Currently implemented recursively. To be changed to a non-recursive
    /// implemented at some point.
    ///
    /// # Examples
    /// ```
    /// use yagraphc::graph::UnGraph;
    /// use yagraphc::graph::traits::{GraphBuilding, Traversable};
    ///
    /// let mut graph = UnGraph::new();
    ///
    /// graph.add_edge(1, 2, ());
    /// graph.add_edge(2, 3, ());
    /// graph.add_edge(3, 4, ());
    /// graph.add_edge(1, 5, ());
    /// graph.add_edge(5, 6, ());
    ///
    /// let dfs = graph.dfs_post_order(1);
    ///
    /// let depths = dfs.map(|(node, _)| node).collect::<Vec<_>>();
    ///
    /// assert!(matches!(depths[..], [6, 5, 4, 3, 2, 1] | [4, 3, 2, 6, 5, 1]));
    // TODO: Implement post-order non-recursively.
    fn dfs_post_order(&self, from: T) -> PostOrderDfsIter<T>
    where
        Self: Sized,
    {
        PostOrderDfsIter::new(self, from)
    }

    /// Finds path from `from` to `to` using BFS.
    ///
    /// Returns `None` if there is no path.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use yagraphc::graph::UnGraph;
    /// use yagraphc::graph::traits::{GraphBuilding, Traversable};
    ///
    /// let mut graph = UnGraph::new();
    ///
    /// graph.add_edge(1, 2, ());
    /// graph.add_edge(2, 3, ());
    /// graph.add_edge(3, 4, ());
    /// graph.add_edge(1, 5, ());
    /// graph.add_edge(5, 6, ());
    ///
    /// let path = graph.find_path(1, 4);
    ///
    /// assert_eq!(path, Some(vec![1, 2, 3, 4]));
    fn find_path(&self, from: T, to: T) -> Option<Vec<T>>
    where
        Self: Sized,
    {
        let mut visited = HashSet::new();
        let mut pairs = HashMap::new();
        let mut queue = VecDeque::new();

        queue.push_back((from, from));

        while let Some((prev, current)) = queue.pop_front() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current);
            pairs.insert(current, prev);

            if current == to {
                let mut node = current;

                let mut path = Vec::new();
                while node != from {
                    path.push(node);

                    node = pairs[&node];
                }

                path.push(from);

                path.reverse();

                return Some(path);
            }

            for (target, _) in self.edges(&current) {
                if visited.contains(&target) {
                    continue;
                }

                queue.push_back((current, target));
            }
        }

        None
    }

    /// Finds path from `from` to `to` using BFS while filtering edges.
    ///
    /// Returns `None` if there is no path.
    ///
    /// For an undirected graph, strive to make predicate(x,y) == predicate(y,x).
    /// As of now, the order is related to the exploration direction of bfs,
    /// but it is advisable not to rely on direction concepts on undirected graphs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use yagraphc::graph::UnGraph;
    /// use yagraphc::graph::traits::{GraphBuilding, Traversable};
    /// let mut graph = UnGraph::default();

    /// graph.add_edge(1, 2, ());
    /// graph.add_edge(2, 3, ());
    /// graph.add_edge(3, 4, ());
    /// graph.add_edge(4, 5, ());

    /// graph.add_edge(1, 7, ());
    /// graph.add_edge(7, 5, ());

    /// let path = graph.find_path(1, 5).unwrap();

    /// assert_eq!(path, vec![1, 7, 5]);

    /// let path = graph
    ///     .find_path_filter_edges(1, 5, |x, y| (x, y) != (1, 7))
    ///     .unwrap();

    /// assert_eq!(path, vec![1, 2, 3, 4, 5]);
    fn find_path_filter_edges<G>(&self, from: T, to: T, predicate: G) -> Option<Vec<T>>
    where
        Self: Sized,
        G: Fn(T, T) -> bool,
    {
        let mut visited = HashSet::new();
        let mut pairs = HashMap::new();
        let mut queue = VecDeque::new();

        queue.push_back((from, from));

        while let Some((prev, current)) = queue.pop_front() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current);
            pairs.insert(current, prev);

            if current == to {
                let mut node = current;

                let mut path = Vec::new();
                while node != from {
                    path.push(node);

                    node = pairs[&node];
                }

                path.push(from);

                path.reverse();

                return Some(path);
            }

            for (target, _) in self.edges(&current) {
                if visited.contains(&target) || !predicate(current, target) {
                    continue;
                }

                queue.push_back((current, target));
            }
        }

        None
    }

    /// Returns a list of connected components of the graph.
    ///
    /// If being used in a directed graph, those are the strongly connected components,
    /// computed using Kosaraju's algorithm.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use yagraphc::graph::{UnGraph, DiGraph};
    /// use yagraphc::graph::traits::{GraphBuilding, Traversable};
    ///
    /// let mut graph = UnGraph::new();
    ///
    /// graph.add_edge(1, 2, ());
    /// graph.add_edge(2, 3, ());
    ///
    /// graph.add_edge(4, 5, ());
    /// graph.add_edge(5, 6, ());
    /// graph.add_edge(6, 4, ());
    ///
    /// let components = graph.connected_components();
    ///
    /// assert_eq!(components.len(), 2);
    ///
    /// let mut graph = DiGraph::new();
    ///
    /// graph.add_edge(1, 2, ());
    /// graph.add_edge(2, 3, ());
    ///
    /// graph.add_edge(4, 5, ());
    /// graph.add_edge(5, 6, ());
    /// graph.add_edge(6, 4, ());
    ///
    /// let components = graph.connected_components();
    ///
    /// assert_eq!(components.len(), 4);
    fn connected_components(&self) -> Vec<Vec<T>>
    where
        Self: Sized,
    {
        let mut visited = HashSet::new();
        let mut stack = Vec::new();

        for node in self.nodes() {
            if visited.contains(&node) {
                continue;
            }

            for (inner_node, _) in self.dfs_post_order(node) {
                visited.insert(inner_node);
                stack.push(inner_node);
            }
        }

        stack.reverse();

        let mut visited = HashSet::new();
        let mut components = Vec::new();

        for node in stack {
            if visited.contains(&node) {
                continue;
            }

            let mut component = Vec::new();

            let mut stack = Vec::new();
            stack.push(node);

            while let Some(node) = stack.pop() {
                if visited.contains(&node) {
                    continue;
                }

                component.push(node);
                visited.insert(node);

                for (inner_node, _) in self.in_edges(&node) {
                    stack.push(inner_node);
                }
            }

            components.push(component);
        }

        components
    }
}

fn copy_tuple<T, W>(x: (&T, &W)) -> (T, W)
where
    T: Clone + Copy,
    W: Clone + Copy,
{
    (*x.0, *x.1)
}

struct QueueEntry<T, W>
where
    T: Clone + Copy,
    W: Ord + PartialOrd,
{
    pub node: T,
    pub cur_cost: W,
}

impl<T, W> Ord for QueueEntry<T, W>
where
    T: Clone + Copy,
    W: Ord + PartialOrd,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.cur_cost.cmp(&self.cur_cost)
    }
}

impl<T, W> PartialOrd for QueueEntry<T, W>
where
    T: Clone + Copy,
    W: Ord + PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T, W> PartialEq for QueueEntry<T, W>
where
    T: Clone + Copy,
    W: Ord + PartialOrd,
{
    fn eq(&self, other: &Self) -> bool {
        self.cur_cost.eq(&other.cur_cost)
    }
}

impl<T, W> Eq for QueueEntry<T, W>
where
    T: Clone + Copy,
    W: Ord + PartialOrd,
{
}

pub trait ArithmeticallyWeightedGraph<T, W>
where
    T: Clone + Copy + Eq + Hash + PartialEq,
    W: Clone
        + Copy
        + std::ops::Add<Output = W>
        + std::ops::Sub<Output = W>
        + PartialOrd
        + Ord
        + Default,
{
    /// Returns the shortest length among paths from `from` to `to`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use yagraphc::graph::UnGraph;
    /// use yagraphc::graph::traits::{GraphBuilding, Traversable};
    /// use yagraphc::graph::traits::ArithmeticallyWeightedGraph;
    ///
    /// let mut graph = UnGraph::new();
    ///
    /// graph.add_edge(1, 2, 1);
    /// graph.add_edge(2, 3, 2);
    /// graph.add_edge(3, 4, 3);
    /// graph.add_edge(1, 4, 7);
    ///
    /// assert_eq!(graph.dijkstra(1, 4), Some(6));
    /// assert_eq!(graph.dijkstra(1, 5), None);
    fn dijkstra(&self, from: T, to: T) -> Option<W>
    where
        Self: Traversable<T, W>,
    {
        let mut visited = HashSet::new();
        let mut distances = HashMap::new();

        distances.insert(from, W::default());

        let mut queue = BinaryHeap::new();
        queue.push(QueueEntry {
            node: from,
            cur_cost: W::default(),
        });

        while let Some(QueueEntry {
            node,
            cur_cost: cur_dist,
        }) = queue.pop()
        {
            if visited.contains(&node) {
                continue;
            }

            if node == to {
                return Some(cur_dist);
            }

            for (target, weight) in self.edges(&node) {
                let mut distance = cur_dist + weight;

                distances
                    .entry(target)
                    .and_modify(|dist| {
                        let best_dist = (*dist).min(cur_dist + weight);

                        distance = best_dist;
                        *dist = best_dist;
                    })
                    .or_insert(cur_dist + weight);

                if !visited.contains(&target) {
                    queue.push(QueueEntry {
                        node: target,
                        cur_cost: distance,
                    })
                }
            }

            visited.insert(node);
        }

        None
    }

    /// Returns the shortest path among paths from `from` to `to`, together with its length.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use yagraphc::graph::UnGraph;
    /// use yagraphc::graph::traits::{GraphBuilding, Traversable};
    /// use yagraphc::graph::traits::ArithmeticallyWeightedGraph;
    ///
    /// let mut graph = UnGraph::new();
    ///
    /// graph.add_edge(1, 2, 1);
    /// graph.add_edge(2, 3, 2);
    /// graph.add_edge(3, 4, 3);
    /// graph.add_edge(1, 4, 7);
    ///
    /// assert_eq!(graph.dijkstra_with_path(1, 4).unwrap().0, vec![1, 2, 3, 4]);
    /// assert_eq!(graph.dijkstra_with_path(1, 5), None);
    fn dijkstra_with_path(&self, from: T, to: T) -> Option<(Vec<T>, W)>
    where
        Self: Traversable<T, W>,
    {
        let mut visited = HashSet::new();
        let mut distances = HashMap::new();

        distances.insert(from, (W::default(), from));

        let mut queue = BinaryHeap::new();
        queue.push(QueueEntry {
            node: from,
            cur_cost: W::default(),
        });

        while let Some(QueueEntry {
            node,
            cur_cost: cur_dist,
        }) = queue.pop()
        {
            if visited.contains(&node) {
                continue;
            }

            if node == to {
                let mut path = Vec::new();

                let mut node = node;

                while node != from {
                    path.push(node);

                    node = distances[&node].1;
                }

                path.push(from);

                path.reverse();

                return Some((path, cur_dist));
            }

            for (target, weight) in self.edges(&node) {
                distances
                    .entry(target)
                    .and_modify(|(dist, previous)| {
                        if cur_dist + weight < *dist {
                            *dist = cur_dist + weight;
                            *previous = node;
                        }
                    })
                    .or_insert((cur_dist + weight, node));

                if !visited.contains(&target) {
                    queue.push(QueueEntry {
                        node: target,
                        cur_cost: distances[&target].0,
                    })
                }
            }

            visited.insert(node);
        }

        None
    }

    /// Returns the shortest length among paths from `from` to `to` using A*.
    ///
    /// `heuristic` corresponds to the heuristic function of the A* algorithm.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use yagraphc::graph::UnGraph;
    /// use yagraphc::graph::traits::{GraphBuilding, Traversable};
    /// use yagraphc::graph::traits::ArithmeticallyWeightedGraph;
    ///
    /// let mut graph = UnGraph::new();
    ///
    /// graph.add_edge(1, 2, 1);
    /// graph.add_edge(2, 3, 2);
    /// graph.add_edge(3, 4, 3);
    /// graph.add_edge(1, 4, 7);
    ///
    /// assert_eq!(graph.a_star(1, 4, |_| 0), Some(6));
    /// assert_eq!(graph.a_star(1, 5, |_| 0), None);
    fn a_star<G>(&self, from: T, to: T, heuristic: G) -> Option<W>
    where
        Self: Traversable<T, W>,
        G: Fn(T) -> W,
    {
        let mut visited = HashSet::new();
        let mut distances = HashMap::new();

        distances.insert(from, W::default());

        let mut queue = BinaryHeap::new();
        queue.push(QueueEntry {
            node: from,
            cur_cost: W::default() + heuristic(from),
        });

        while let Some(QueueEntry { node, .. }) = queue.pop() {
            if visited.contains(&node) {
                continue;
            }

            if node == to {
                return Some(distances[&node]);
            }

            for (target, weight) in self.edges(&node) {
                let mut distance = distances[&node] + weight;

                distances
                    .entry(target)
                    .and_modify(|dist| {
                        let best_dist = (*dist).min(distance);

                        distance = best_dist;
                        *dist = best_dist;
                    })
                    .or_insert(distance);

                if !visited.contains(&target) {
                    queue.push(QueueEntry {
                        node: target,
                        cur_cost: distance + heuristic(target),
                    })
                }
            }

            visited.insert(node);
        }

        None
    }

    /// Returns the shortest path from `from` to `to` using A*, together with its length.
    ///
    /// `heuristic` corresponds to the heuristic function of the A* algorithm.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use yagraphc::graph::UnGraph;
    /// use yagraphc::graph::traits::{GraphBuilding, Traversable};
    /// use yagraphc::graph::traits::ArithmeticallyWeightedGraph;
    ///
    /// let mut graph = UnGraph::new();
    ///
    /// graph.add_edge(1, 2, 1);
    /// graph.add_edge(2, 3, 2);
    /// graph.add_edge(3, 4, 3);
    /// graph.add_edge(1, 4, 7);
    ///
    /// assert_eq!(graph.a_star_with_path(1, 4, |_| 0).unwrap().0, vec![1, 2, 3, 4]);
    /// assert_eq!(graph.a_star_with_path(1, 5, |_| 0), None);
    fn a_star_with_path<G>(&self, from: T, to: T, heuristic: G) -> Option<(Vec<T>, W)>
    where
        Self: Traversable<T, W>,
        G: Fn(T) -> W,
    {
        let mut visited = HashSet::new();
        let mut distances = HashMap::new();

        distances.insert(from, (W::default(), from));

        let mut queue = BinaryHeap::new();
        queue.push(QueueEntry {
            node: from,
            cur_cost: W::default() + heuristic(from),
        });

        while let Some(QueueEntry { node, .. }) = queue.pop() {
            if visited.contains(&node) {
                continue;
            }

            if node == to {
                let mut path = Vec::new();

                let mut node = node;

                while node != from {
                    path.push(node);

                    node = distances[&node].1;
                }

                path.push(from);

                path.reverse();

                return Some((path, distances[&node].0));
            }

            for (target, weight) in self.edges(&node) {
                let mut distance = distances[&node].0 + weight;

                distances
                    .entry(target)
                    .and_modify(|(dist, prev)| {
                        if distance < *dist {
                            *dist = distance;
                            *prev = node;
                        } else {
                            distance = *dist
                        }
                    })
                    .or_insert((distance, node));

                if !visited.contains(&target) {
                    queue.push(QueueEntry {
                        node: target,
                        cur_cost: distance + heuristic(target),
                    })
                }
            }

            visited.insert(node);
        }

        None
    }

    /// Runs the Edmonds-Karp algorithm on the graph to find max flow.
    ///
    /// Assumes the edge weights are the capacities.
    ///
    /// Returns a HashMap with the flow values for each edge.
    ///
    /// Please select a number type for W which allows for subtraction
    /// and negative values, otherwise there may be undefined behavior.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use yagraphc::graph::UnGraph;
    /// use yagraphc::graph::traits::{ArithmeticallyWeightedGraph, GraphBuilding, Traversable};
    ///
    /// let mut graph = UnGraph::new();
    /// graph.add_edge(1, 2, 1000);
    /// graph.add_edge(2, 4, 1000);
    /// graph.add_edge(1, 3, 1000);
    /// graph.add_edge(3, 4, 1000);
    ///
    /// graph.add_edge(2, 3, 1);
    ///
    /// let flows = graph.edmonds_karp(1, 4);
    /// assert_eq!(*flows.get(&(1, 2)).unwrap(), 1000);
    /// assert_eq!(*flows.get(&(1, 3)).unwrap(), 1000);
    ///
    /// assert_eq!(*flows.get(&(2, 3)).unwrap_or(&0), 0);
    fn edmonds_karp(&self, source: T, sink: T) -> HashMap<(T, T), W>
    where
        Self: Traversable<T, W> + Sized,
    {
        let flows = HashMap::new();

        let mut residual_obtension = ResidualNetwork { flows, graph: self };

        while let Some(path) = self.find_path_filter_edges(source, sink, |x, y| {
            residual_obtension.get_residual_capacity(x, y) > W::default()
        }) {
            let residuals_in_path = path
                .iter()
                .zip(&path[1..])
                .map(|(&x, &y)| residual_obtension.get_residual_capacity(x, y));

            let min_res = residuals_in_path.min().expect("Path should not be empty");

            path.iter().zip(&path[1..]).for_each(|(&x, &y)| {
                residual_obtension
                    .flows
                    .entry((x, y))
                    .and_modify(|v| *v = *v + min_res)
                    .or_insert(min_res);
                residual_obtension
                    .flows
                    .entry((y, x))
                    .and_modify(|v| *v = *v - min_res)
                    .or_insert(W::default() - min_res);
            });

            residual_obtension = ResidualNetwork {
                flows: residual_obtension.flows,
                graph: self,
            };
        }

        residual_obtension.flows
    }
}

struct ResidualNetwork<'a, T, W> {
    flows: HashMap<(T, T), W>,
    graph: &'a dyn Traversable<T, W>,
}

impl<'a, T, W> ResidualNetwork<'a, T, W>
where
    T: Clone + Copy + Eq + Hash + PartialEq,
    W: Clone
        + Copy
        + std::ops::Add<Output = W>
        + std::ops::Sub<Output = W>
        + PartialOrd
        + Ord
        + Default,
{
    fn get_residual_capacity(&self, s: T, t: T) -> W {
        self.graph
            .edge_weight(s, t)
            .expect("Should only be considering existing edges")
            - self.flows.get(&(s, t)).copied().unwrap_or(W::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error() {
        let e = NodeNotFound;
        dbg!(&e);
    }

    #[test]
    fn test_queue_entry() {
        let queue_entry1 = QueueEntry {
            node: 1,
            cur_cost: 12,
        };
        let queue_entry2 = QueueEntry {
            node: 4,
            cur_cost: 12,
        };

        assert!(queue_entry1 == queue_entry2);
    }
}
