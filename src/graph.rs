use anyhow::{anyhow, Result};

use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::hash::Hash;

use self::traits::ArithmeticallyWeightedGraph;
use self::traits::Graph;

pub mod traits;

/// Undirected graph using adjancency list as a nested HashMap.
///
/// Recommended for graphs that have vertices with high degree.
#[derive(Debug, Clone)]
pub struct UnGraph<T, W> {
    nodes: HashSet<T>,
    edges: HashMap<T, HashMap<T, W>>,

    empty: HashMap<T, W>,
}

impl<T, W> UnGraph<T, W>
where
    T: Clone + Copy + Hash + Eq + PartialEq,
    W: Clone + Copy,
{
    /// Initializes an empty undirected graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Finds basic cycles of the undirected graph.
    ///
    /// Basic cycles correspond to generators of the first homology group of the graph.
    ///
    ///
    /// # Examples
    ///
    /// ```rust
    /// use yagraphc::graph::UnGraph;
    /// use yagraphc::graph::traits::Graph;
    ///
    /// let mut graph = UnGraph::default();
    ///
    /// graph.add_edge(1, 2, ());
    /// graph.add_edge(2, 3, ());
    /// graph.add_edge(3, 4, ());
    /// graph.add_edge(4, 1, ());
    ///
    /// let cycles = graph.basic_cycles();
    /// assert_eq!(cycles.len(), 1);
    /// assert_eq!(cycles[0].len(), 4);
    ///
    /// graph.remove_edge(2, 3);
    /// let cycles = graph.basic_cycles();
    /// assert_eq!(cycles.len(), 0);
    /// ```
    pub fn basic_cycles(&self) -> Vec<Vec<T>> {
        let mut remaining_edges = self.all_edges();

        let mut spanning_forest = UnGraph::default();
        let mut visited = HashSet::new();

        for node in self.nodes() {
            if visited.contains(&node) {
                continue;
            }

            let mut queue = VecDeque::new();
            queue.push_back((node, node));

            while let Some((previous_node, inner_node)) = queue.pop_front() {
                if visited.contains(&inner_node) {
                    continue;
                }

                remaining_edges.remove(&(inner_node, previous_node));
                remaining_edges.remove(&(previous_node, inner_node));
                spanning_forest.add_edge(previous_node, inner_node, 1);

                visited.insert(inner_node);

                for (target, _) in self.edges(&inner_node) {
                    if visited.contains(&target) {
                        continue;
                    }

                    queue.push_back((inner_node, target));
                }
            }
        }

        let mut cycles = Vec::new();

        for edge in remaining_edges {
            if let Some(path) = spanning_forest.find_path(edge.0, edge.1) {
                cycles.push(path);
            }
        }

        cycles
    }

    /// Gets all edges of the graph as a set.
    ///
    /// If edge (a, b) is returned, then (b, a) will not. The ordering of the nodes is random.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use yagraphc::graph::UnGraph;
    /// use yagraphc::graph::traits::Graph;
    ///
    /// let mut graph = UnGraph::default();
    ///
    /// graph.add_edge(1, 2, ());
    /// graph.add_edge(2, 3, ());
    ///
    /// let edges = graph.all_edges();
    /// assert_eq!(edges.len(), 2);
    pub fn all_edges(&self) -> HashSet<(T, T)> {
        let mut edges = HashSet::new();

        for (origin, destinations) in &self.edges {
            for dest in destinations.keys() {
                if edges.contains(&(*dest, *origin)) {
                    continue;
                }
                edges.insert((*origin, *dest));
            }
        }

        edges
    }
}

impl<T, W> Graph<T, W> for UnGraph<T, W>
where
    T: Clone + Copy + Hash + Eq + PartialEq,
    W: Clone + Copy,
{
    fn add_edge(&mut self, from: T, to: T, weight: W) {
        self.edges.entry(from).or_default().insert(to, weight);
        self.edges.entry(to).or_default().insert(from, weight);

        self.nodes.insert(from);
        self.nodes.insert(to);
    }

    fn add_node(&mut self, node: T) -> bool {
        self.nodes.insert(node)
    }

    fn remove_edge(&mut self, from: T, to: T) -> Result<()> {
        self.edges
            .get_mut(&from)
            .ok_or(anyhow!("Node not found"))?
            .remove(&to);
        self.edges
            .get_mut(&to)
            .ok_or(anyhow!("Node not found"))?
            .remove(&from);

        Ok(())
    }

    fn remove_node(&mut self, node: T) -> Result<()> {
        self.nodes.remove(&node);

        let to_nodes: Vec<T> = self
            .edges
            .get(&node)
            .ok_or(anyhow!("Node not found"))?
            .keys()
            .copied()
            .collect();

        for to_node in to_nodes {
            self.edges
                .get_mut(&to_node)
                .ok_or(anyhow!("Node not found"))?
                .remove(&node);
        }

        Ok(())
    }

    fn nodes(&self) -> traits::NodeIter<'_, T> {
        traits::NodeIter {
            nodes_iter: self.nodes.iter(),
        }
    }

    fn edges(&self, n: &T) -> traits::EdgeIterType<T, W> {
        let edges = self.edges.get(n);

        if let Some(edges) = edges {
            traits::EdgeIterType::EdgeIter(traits::EdgeIter {
                edge_iter: edges.iter(),
            })
        } else {
            traits::EdgeIterType::EdgeIter(traits::EdgeIter {
                edge_iter: self.empty.iter(),
            })
        }
    }

    fn in_edges(&self, n: &T) -> traits::EdgeIterType<T, W> {
        self.edges(n)
    }

    fn has_edge(&self, from: T, to: T) -> bool {
        self.edges
            .get(&from)
            .map_or(false, |edges| edges.contains_key(&to))
    }

    /// Computes the connected components of an undirected graph.
    ///
    /// Relies simply on BFS to compute the components as opposed to Kosaraju's algorithm.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use yagraphc::graph::UnGraph;
    /// use yagraphc::graph::traits::Graph;
    ///
    /// let mut graph = UnGraph::default();
    ///
    /// graph.add_edge(1, 2, ());
    /// graph.add_edge(2, 3, ());
    /// graph.add_edge(3, 4, ());
    ///
    /// let components = graph.connected_components();
    /// assert_eq!(components.len(), 1);
    ///
    /// graph.add_edge(5, 6, ());
    ///
    /// let components = graph.connected_components();
    /// assert_eq!(components.len(), 2);
    fn connected_components(&self) -> Vec<Vec<T>>
    where
        Self: Sized,
    {
        let mut visited = HashSet::new();
        let mut components = vec![];

        for node in self.nodes() {
            if visited.contains(&node) {
                continue;
            }

            let mut component = vec![node];

            for (inner_node, _) in self.bfs(node) {
                if visited.contains(&inner_node) {
                    continue;
                }

                component.push(inner_node);
                visited.insert(inner_node);
            }

            components.push(component);
        }

        components
    }
}

impl<T, W> ArithmeticallyWeightedGraph<T, W> for UnGraph<T, W>
where
    T: Clone + Copy + Eq + Hash + PartialEq,
    W: Clone + Copy + std::ops::Add<Output = W> + PartialOrd + Ord + Default,
{
}

impl<T, W> Default for UnGraph<T, W> {
    fn default() -> Self {
        UnGraph {
            nodes: HashSet::new(),
            edges: HashMap::new(),
            empty: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct UnGraphVecEdges<T, W> {
    nodes: HashSet<T>,
    edges: HashMap<T, Vec<(T, W)>>,

    empty: Vec<(T, W)>,
}

impl<T, W> UnGraphVecEdges<T, W>
where
    T: Clone + Copy + Hash + Eq + PartialEq,
    W: Clone + Copy,
{
    pub fn new() -> Self {
        Self::default()
    }

    /// Finds basic cycles of the undirected graph.
    ///
    /// Basic cycles correspond to generators of the first homology group of the graph.
    ///
    ///
    /// # Examples
    ///
    /// ```rust
    /// use yagraphc::graph::UnGraph;
    /// use yagraphc::graph::traits::Graph;
    ///
    /// let mut graph = UnGraph::default();
    ///
    /// graph.add_edge(1, 2, ());
    /// graph.add_edge(2, 3, ());
    /// graph.add_edge(3, 4, ());
    /// graph.add_edge(4, 1, ());
    ///
    /// let cycles = graph.basic_cycles();
    /// assert_eq!(cycles.len(), 1);
    /// assert_eq!(cycles[0].len(), 4);
    ///
    /// graph.remove_edge(2, 3);
    /// let cycles = graph.basic_cycles();
    /// assert_eq!(cycles.len(), 0);
    /// ```
    pub fn basic_cycles(&self) -> Vec<Vec<T>> {
        let mut remaining_edges = self.all_edges();

        let mut spanning_forest = UnGraph::default();
        let mut visited = HashSet::new();

        for node in self.nodes() {
            if visited.contains(&node) {
                continue;
            }

            let mut queue = VecDeque::new();
            queue.push_back((node, node));

            while let Some((previous_node, inner_node)) = queue.pop_front() {
                if visited.contains(&inner_node) {
                    continue;
                }

                remaining_edges.remove(&(inner_node, previous_node));
                remaining_edges.remove(&(previous_node, inner_node));
                spanning_forest.add_edge(previous_node, inner_node, 1);

                visited.insert(inner_node);

                for (target, _) in self.edges(&inner_node) {
                    if visited.contains(&target) {
                        continue;
                    }

                    queue.push_back((inner_node, target));
                }
            }
        }

        let mut cycles = Vec::new();

        for edge in remaining_edges {
            if let Some(path) = spanning_forest.find_path(edge.0, edge.1) {
                cycles.push(path);
            }
        }

        cycles
    }

    /// Gets all edges of the graph as a set.
    ///
    /// If edge (a, b) is returned, then (b, a) will not. The ordering of the nodes is random.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use yagraphc::graph::UnGraphVecEdges;
    /// use yagraphc::graph::traits::Graph;
    ///
    /// let mut graph = UnGraphVecEdges::default();
    ///
    /// graph.add_edge(1, 2, ());
    /// graph.add_edge(2, 3, ());
    ///
    /// let edges = graph.all_edges();
    /// assert_eq!(edges.len(), 2);
    pub fn all_edges(&self) -> HashSet<(T, T)> {
        let mut edges = HashSet::new();

        for (origin, destinations) in &self.edges {
            for (dest, _) in destinations {
                if edges.contains(&(*dest, *origin)) {
                    continue;
                }
                edges.insert((*origin, *dest));
            }
        }

        edges
    }
}

impl<T, W> Graph<T, W> for UnGraphVecEdges<T, W>
where
    T: Clone + Copy + Hash + Eq + PartialEq,
    W: Clone + Copy,
{
    fn add_edge(&mut self, from: T, to: T, weight: W) {
        self.edges.entry(from).or_default().push((to, weight));
        self.edges.entry(to).or_default().push((from, weight));

        self.nodes.insert(from);
        self.nodes.insert(to);
    }

    fn add_node(&mut self, node: T) -> bool {
        self.nodes.insert(node)
    }

    fn remove_edge(&mut self, from: T, to: T) -> Result<()> {
        let edges_beginning_at_from = self.edges.get_mut(&from).ok_or(anyhow!("Node not found"))?;

        edges_beginning_at_from.retain(|(target, _)| *target != to);

        let edges_beginning_at_to = self.edges.get_mut(&to).ok_or(anyhow!("Node not found"))?;

        edges_beginning_at_to.retain(|(target, _)| *target != from);

        Ok(())
    }

    fn remove_node(&mut self, node: T) -> Result<()> {
        self.nodes.remove(&node);

        let to_nodes = self
            .edges
            .get(&node)
            .ok_or(anyhow!("Node not found"))?
            .to_vec();

        for (to_node, _) in to_nodes {
            self.edges
                .get_mut(&to_node)
                .ok_or(anyhow!("Node not found"))?
                .retain(|(target, _)| *target != node);
        }

        Ok(())
    }

    fn nodes(&self) -> traits::NodeIter<'_, T> {
        traits::NodeIter {
            nodes_iter: self.nodes.iter(),
        }
    }

    fn edges(&self, n: &T) -> traits::EdgeIterType<T, W> {
        let edges = self.edges.get(n);

        if let Some(edges) = edges {
            traits::EdgeIterType::EdgeIterVec(traits::EdgeIterVec {
                edge_iter: edges.iter(),
            })
        } else {
            traits::EdgeIterType::EdgeIterVec(traits::EdgeIterVec {
                edge_iter: self.empty.iter(),
            })
        }
    }

    fn in_edges(&self, n: &T) -> traits::EdgeIterType<T, W> {
        self.edges(n)
    }

    fn has_edge(&self, from: T, to: T) -> bool {
        self.edges
            .get(&from)
            .map_or(false, |edges| edges.iter().any(|(target, _)| *target == to))
    }

    /// Computes the connected components of an undirected graph.
    ///
    /// Relies simply on BFS to compute the components as opposed to Kosaraju's algorithm.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use yagraphc::graph::UnGraphVecEdges;
    /// use yagraphc::graph::traits::Graph;
    ///
    /// let mut graph = UnGraphVecEdges::default();
    ///
    /// graph.add_edge(1, 2, ());
    /// graph.add_edge(2, 3, ());
    /// graph.add_edge(3, 4, ());
    ///
    /// let components = graph.connected_components();
    /// assert_eq!(components.len(), 1);
    ///
    /// graph.add_edge(5, 6, ());
    ///
    /// let components = graph.connected_components();
    /// assert_eq!(components.len(), 2);
    fn connected_components(&self) -> Vec<Vec<T>>
    where
        Self: Sized,
    {
        let mut visited = HashSet::new();
        let mut components = vec![];

        for node in self.nodes() {
            if visited.contains(&node) {
                continue;
            }

            let mut component = vec![node];

            for (inner_node, _) in self.bfs(node) {
                if visited.contains(&inner_node) {
                    continue;
                }

                component.push(inner_node);
                visited.insert(inner_node);
            }

            components.push(component);
        }

        components
    }
}

impl<T, W> ArithmeticallyWeightedGraph<T, W> for UnGraphVecEdges<T, W>
where
    T: Clone + Copy + Eq + Hash + PartialEq,
    W: Clone + Copy + std::ops::Add<Output = W> + PartialOrd + Ord + Default,
{
}

impl<T, W> Default for UnGraphVecEdges<T, W> {
    fn default() -> Self {
        UnGraphVecEdges {
            nodes: HashSet::new(),
            edges: HashMap::new(),
            empty: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DiGraph<T, W> {
    nodes: HashSet<T>,
    edges: HashMap<T, HashMap<T, W>>,
    in_edges: HashMap<T, HashMap<T, W>>,

    empty: HashMap<T, W>,
}

impl<T, W> Default for DiGraph<T, W> {
    fn default() -> Self {
        DiGraph {
            nodes: HashSet::new(),
            edges: HashMap::new(),
            in_edges: HashMap::new(),
            empty: HashMap::new(),
        }
    }
}

impl<T, W> DiGraph<T, W> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T, W> Graph<T, W> for DiGraph<T, W>
where
    T: Clone + Copy + Eq + Hash + PartialEq,
    W: Clone + Copy,
{
    fn add_edge(&mut self, from: T, to: T, weight: W) {
        self.edges.entry(from).or_default().insert(to, weight);
        self.in_edges.entry(to).or_default().insert(from, weight);

        self.add_node(from);
        self.add_node(to);
    }

    fn add_node(&mut self, node: T) -> bool {
        self.nodes.insert(node)
    }

    fn remove_edge(&mut self, from: T, to: T) -> Result<()> {
        self.edges
            .get_mut(&from)
            .ok_or(anyhow!("Node not found"))?
            .remove(&to);
        self.in_edges
            .get_mut(&to)
            .ok_or(anyhow!("Node not found"))?
            .remove(&from);

        Ok(())
    }

    fn remove_node(&mut self, node: T) -> Result<()> {
        self.nodes.remove(&node);

        let to_nodes: Vec<T> = self
            .edges
            .get(&node)
            .ok_or(anyhow!("Node not found"))?
            .keys()
            .copied()
            .collect();

        for to_node in to_nodes {
            self.edges
                .get_mut(&to_node)
                .ok_or(anyhow!("Node not found"))?
                .remove(&node);
        }

        Ok(())
    }

    fn nodes(&self) -> traits::NodeIter<'_, T> {
        traits::NodeIter {
            nodes_iter: self.nodes.iter(),
        }
    }

    fn edges(&self, n: &T) -> traits::EdgeIterType<T, W> {
        let edges = self.edges.get(n);

        if let Some(edges) = edges {
            traits::EdgeIterType::EdgeIter(traits::EdgeIter {
                edge_iter: edges.iter(),
            })
        } else {
            traits::EdgeIterType::EdgeIter(traits::EdgeIter {
                edge_iter: self.empty.iter(),
            })
        }
    }

    fn in_edges(&self, n: &T) -> traits::EdgeIterType<T, W> {
        let edges = self.in_edges.get(n);

        if let Some(edges) = edges {
            traits::EdgeIterType::EdgeIter(traits::EdgeIter {
                edge_iter: edges.iter(),
            })
        } else {
            traits::EdgeIterType::EdgeIter(traits::EdgeIter {
                edge_iter: self.empty.iter(),
            })
        }
    }

    fn has_edge(&self, from: T, to: T) -> bool {
        self.edges
            .get(&from)
            .map_or(false, |edges| edges.contains_key(&to))
    }
}

impl<T, W> ArithmeticallyWeightedGraph<T, W> for DiGraph<T, W>
where
    T: Clone + Copy + Eq + Hash + PartialEq,
    W: Clone + Copy + std::ops::Add<Output = W> + PartialOrd + Ord + Default,
{
}

#[derive(Debug, Clone)]
pub struct DiGraphVecEdges<T, W> {
    nodes: HashSet<T>,
    edges: HashMap<T, Vec<(T, W)>>,
    in_edges: HashMap<T, Vec<(T, W)>>,

    empty: Vec<(T, W)>,
}

impl<T, W> Default for DiGraphVecEdges<T, W> {
    fn default() -> Self {
        DiGraphVecEdges {
            nodes: HashSet::new(),
            edges: HashMap::new(),
            in_edges: HashMap::new(),
            empty: Vec::new(),
        }
    }
}

impl<T, W> DiGraphVecEdges<T, W> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T, W> Graph<T, W> for DiGraphVecEdges<T, W>
where
    T: Clone + Copy + Eq + Hash + PartialEq,
    W: Clone + Copy,
{
    fn add_edge(&mut self, from: T, to: T, weight: W) {
        self.edges.entry(from).or_default().push((to, weight));
        self.edges.entry(to).or_default().push((from, weight));
    }

    fn add_node(&mut self, node: T) -> bool {
        self.nodes.insert(node)
    }

    fn remove_edge(&mut self, from: T, to: T) -> Result<()> {
        self.edges
            .get_mut(&from)
            .ok_or(anyhow!("Node not found"))?
            .retain(|(node, _)| *node != to);

        self.in_edges
            .get_mut(&to)
            .ok_or(anyhow!("Node not found"))?
            .retain(|(node, _)| *node != from);

        Ok(())
    }

    fn remove_node(&mut self, node: T) -> Result<()> {
        self.nodes.remove(&node);

        let to_nodes: Vec<(T, W)> = self
            .edges
            .get(&node)
            .ok_or(anyhow!("Node not found"))?
            .to_vec();

        for (to_node, _) in to_nodes {
            self.edges
                .get_mut(&to_node)
                .ok_or(anyhow!("Node not found"))?
                .retain(|(n, _)| *n != node);
        }

        Ok(())
    }

    fn nodes(&self) -> traits::NodeIter<'_, T> {
        traits::NodeIter {
            nodes_iter: self.nodes.iter(),
        }
    }

    fn edges(&self, n: &T) -> traits::EdgeIterType<T, W> {
        let edges = self.edges.get(n);

        if let Some(edges) = edges {
            traits::EdgeIterType::EdgeIterVec(traits::EdgeIterVec {
                edge_iter: edges.iter(),
            })
        } else {
            traits::EdgeIterType::EdgeIterVec(traits::EdgeIterVec {
                edge_iter: self.empty.iter(),
            })
        }
    }

    fn in_edges(&self, n: &T) -> traits::EdgeIterType<T, W> {
        let edges = self.in_edges.get(n);

        if let Some(edges) = edges {
            traits::EdgeIterType::EdgeIterVec(traits::EdgeIterVec {
                edge_iter: edges.iter(),
            })
        } else {
            traits::EdgeIterType::EdgeIterVec(traits::EdgeIterVec {
                edge_iter: self.empty.iter(),
            })
        }
    }

    fn has_edge(&self, from: T, to: T) -> bool {
        self.edges
            .get(&from)
            .map_or(false, |edges| edges.iter().any(|(node, _)| *node == to))
    }
}

impl<T, W> ArithmeticallyWeightedGraph<T, W> for DiGraphVecEdges<T, W>
where
    T: Clone + Copy + Eq + Hash + PartialEq,
    W: Clone + Copy + std::ops::Add<Output = W> + PartialOrd + Ord + Default,
{
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Clone, Copy, Default)]
    struct Finite(f64);

    impl Ord for Finite {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.0
                .partial_cmp(&other.0)
                .expect("Should be finite values")
        }
    }

    impl PartialOrd for Finite {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Eq for Finite {}

    impl std::ops::Add for Finite {
        type Output = Self;
        fn add(self, rhs: Self) -> Self::Output {
            Self(self.0 + rhs.0)
        }
    }

    #[test]
    fn test_dijkstra() {
        let mut graph = UnGraph::default();

        graph.add_edge(1, 2, 3);
        graph.add_edge(2, 3, 10);
        graph.add_edge(1, 3, 15);

        assert_eq!(graph.dijkstra(1, 3), Some(13));
        assert_eq!(graph.dijkstra_with_path(1, 3).unwrap().0, vec![1, 2, 3]);

        graph.add_edge(1, 4, 7);
        graph.add_edge(4, 3, 5);

        assert_eq!(graph.dijkstra(1, 3), Some(12));
        assert_eq!(graph.dijkstra_with_path(1, 3).unwrap().0, vec![1, 4, 3]);
    }

    #[test]
    fn test_dijkstra_directed() {
        let mut graph = DiGraph::default();

        graph.add_edge(1, 2, 3);
        graph.add_edge(2, 3, 10);
        graph.add_edge(1, 3, 15);

        assert_eq!(graph.dijkstra(1, 3), Some(13));
        assert_eq!(graph.dijkstra(3, 1), None);

        graph.add_edge(1, 4, 7);
        graph.add_edge(4, 3, 5);

        assert_eq!(graph.dijkstra(1, 3), Some(12));
        assert_eq!(graph.dijkstra(3, 1), None);
    }

    #[test]
    fn test_bfs() {
        let mut graph = UnGraph::default();

        graph.add_edge(1, 2, 3);
        graph.add_edge(2, 3, 10);

        assert_eq!(graph.bfs(1).find(|x| x.0 == 3).unwrap(), (3, 2));

        graph.add_edge(1, 3, 15);

        assert_eq!(graph.bfs(1).find(|x| x.0 == 3), Some((3, 1)));
    }
    #[test]

    fn test_bfs2() {
        let mut graph = UnGraph::default();

        graph.add_edge(1, 2, 3);
        graph.add_edge(2, 3, 10);
        graph.add_edge(1, 3, 15);
        graph.add_edge(3, 4, 4);
        graph.add_edge(4, 5, 7);
        graph.add_edge(1, 10, 3);

        let values: Vec<(i32, usize)> = graph.bfs(1).collect();

        assert_eq!(values.len(), 6);
    }

    #[test]
    fn test_dijkstra_str() {
        let mut graph = DiGraph::default();

        graph.add_edge("a", "b", Finite(12.3));
        graph.add_edge("b", "c", Finite(4.3));
        graph.add_edge("c", "d", Finite(6.2));

        graph.add_edge("a", "d", Finite(25.3));

        assert_eq!(graph.dijkstra("a", "d").unwrap(), Finite(22.8));
    }

    #[test]
    fn test_a_star() {
        let mut graph = DiGraph::default();

        graph.add_edge(1, 2, 12);
        graph.add_edge(2, 3, 13);
        graph.add_edge(3, 4, 8);

        graph.add_edge(1, 4, 40);

        assert_eq!(graph.a_star(1, 4, |x| 4 - x).unwrap(), 33);
    }

    #[test]
    fn test_a_star_with_path() {
        let mut graph = DiGraph::default();

        graph.add_edge(1, 2, 12);
        graph.add_edge(2, 3, 13);
        graph.add_edge(3, 4, 8);

        graph.add_edge(1, 4, 40);

        assert_eq!(
            graph.a_star_with_path(1, 4, |x| 4 - x).unwrap().0,
            vec![1, 2, 3, 4]
        );
    }

    #[test]
    fn test_connected_components() {
        let mut graph = DiGraph::default();

        graph.add_edge(1, 2, 12);
        graph.add_edge(2, 3, 13);
        graph.add_edge(3, 4, 8);

        graph.add_edge(5, 6, 40);

        assert_eq!(graph.connected_components().len(), 6);

        graph.add_edge(4, 1, 8);

        assert_eq!(graph.connected_components().len(), 3);

        let mut graph = UnGraph::default();

        graph.add_edge(1, 2, 12);
        graph.add_edge(2, 3, 13);
        graph.add_edge(3, 4, 8);

        graph.add_edge(5, 6, 40);

        assert_eq!(graph.connected_components().len(), 2);
    }

    #[test]
    fn test_connected_components2() {
        let mut graph = DiGraph::default();

        for i in 0..15 {
            for j in (i + 1)..15 {
                graph.add_edge(i, j, ())
            }
        }

        assert_eq!(graph.connected_components().len(), 15);

        let mut graph = UnGraph::default();

        for i in 0..15 {
            for j in (i + 1)..15 {
                graph.add_edge(i, j, ())
            }
        }

        assert_eq!(graph.connected_components().len(), 1);
    }

    #[test]
    fn test_dfs_post_order() {
        let mut graph = UnGraph::default();

        graph.add_edge(1, 2, ());
        graph.add_edge(1, 3, ());

        graph.add_edge(2, 4, ());
        graph.add_edge(2, 5, ());

        graph.add_edge(3, 6, ());
        graph.add_edge(6, 7, ());
        graph.add_edge(7, 8, ());

        graph.add_edge(5, 9, ());
        graph.add_edge(5, 10, ());

        let dfs_post_order: Vec<_> = graph.dfs_post_order(1).map(|x| x.0).collect();

        assert_eq!(dfs_post_order.len(), 10);
    }

    #[test]
    fn test_cycles() {
        let mut graph = UnGraph::default();

        graph.add_edge(1, 2, ());
        graph.add_edge(2, 3, ());
        graph.add_edge(3, 4, ());

        let cycles = graph.basic_cycles();
        assert_eq!(cycles.len(), 0);

        graph.add_edge(4, 1, ());
        let cycles = graph.basic_cycles();
        assert_eq!(cycles.len(), 1);
    }

    #[test]
    fn test_cycles2() {
        let mut graph = UnGraph::default();

        graph.add_edge(1, 2, ());
        graph.add_edge(2, 3, ());
        graph.add_edge(3, 4, ());
        graph.add_edge(4, 1, ());

        graph.add_edge(2, 5, ());

        graph.add_edge(5, 6, ());
        graph.add_edge(6, 7, ());
        graph.add_edge(7, 5, ());

        let cycles = graph.basic_cycles();
        assert_eq!(cycles.len(), 2);

        graph.remove_edge(6, 7).unwrap();

        let cycles = graph.basic_cycles();
        assert_eq!(cycles.len(), 1);
    }
}
