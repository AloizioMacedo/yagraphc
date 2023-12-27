use anyhow::{anyhow, Result};

use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::hash::Hash;

pub struct NodeIter<'a, T> {
    nodes_iter: std::collections::hash_set::Iter<'a, T>,
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

pub struct EdgeIter<'a, T, W> {
    edge_iter: std::collections::hash_map::Iter<'a, T, W>,
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

pub trait Graph<T, W>
where
    T: Clone + Copy + Eq + Hash + PartialEq,
    W: Clone + Copy,
{
    fn add_edge(&mut self, from: T, to: T, weight: W);

    fn add_node(&mut self, node: T) -> bool;

    fn remove_edge(&mut self, from: T, to: T) -> Result<()>;

    fn remove_node(&mut self, node: T) -> Result<()>;

    fn edges(&self, n: &T) -> EdgeIter<T, W>;

    fn nodes(&self) -> NodeIter<T>;

    fn bfs(&self, from: T, condition: impl Fn(&T) -> bool) -> Option<(T, usize)> {
        let mut visited = HashSet::new();

        let mut queue = VecDeque::new();
        queue.push_back((from, 0));

        while let Some((node, depth)) = queue.pop_front() {
            visited.insert(node);

            for (next, _) in self.edges(&node) {
                if condition(&next) {
                    return Some((next, depth + 1));
                }

                if visited.contains(&next) {
                    continue;
                } else {
                    queue.push_front((next, depth + 1))
                }
            }
        }

        None
    }

    fn dfs(&self, from: T, condition: impl Fn(&T) -> bool) -> Option<(T, usize)> {
        let mut visited = HashSet::new();

        let mut queue = VecDeque::new();
        queue.push_back((from, 0));

        while let Some((node, depth)) = queue.pop_back() {
            visited.insert(node);

            for (next, _) in self.edges(&node) {
                if condition(&next) {
                    return Some((next, depth + 1));
                }

                if visited.contains(&next) {
                    continue;
                } else {
                    queue.push_front((next, depth + 1))
                }
            }
        }

        None
    }
}

struct QueueEntry<T, W>
where
    T: Clone + Copy,
    W: Ord + PartialOrd,
{
    node: T,
    cur_dist: W,
}

impl<T, W> Ord for QueueEntry<T, W>
where
    T: Clone + Copy,
    W: Ord + PartialOrd,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.cur_dist.cmp(&self.cur_dist)
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
        self.cur_dist.eq(&other.cur_dist)
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
    W: Clone + Copy + std::ops::Add<Output = W> + PartialOrd + Ord + Default,
{
    fn dijkstra(&self, from: T, to: T) -> Option<W>
    where
        Self: Graph<T, W>,
    {
        let mut visited = HashSet::new();
        let mut distances = HashMap::new();

        distances.insert(from, W::default());

        let mut queue = BinaryHeap::new();
        queue.push(QueueEntry {
            node: from,
            cur_dist: W::default(),
        });

        while let Some(QueueEntry { node, cur_dist }) = queue.pop() {
            if node == to {
                return Some(cur_dist);
            }

            for (target, weight) in self.edges(&node) {
                distances
                    .entry(target)
                    .and_modify(|dist| {
                        let best_dist = (*dist).min(cur_dist + weight);
                        *dist = best_dist;
                    })
                    .or_insert(cur_dist + weight);

                if !visited.contains(&target) {
                    queue.push(QueueEntry {
                        node: target,
                        cur_dist: distances[&target],
                    })
                }
            }

            visited.insert(node);
        }

        None
    }

    fn dijkstra_with_path(&self, from: T, to: T) -> Option<(Vec<T>, W)>
    where
        Self: Graph<T, W>,
    {
        let mut visited = HashSet::new();
        let mut distances = HashMap::new();

        distances.insert(from, (W::default(), from));

        let mut queue = BinaryHeap::new();
        queue.push(QueueEntry {
            node: from,
            cur_dist: W::default(),
        });

        while let Some(QueueEntry { node, cur_dist }) = queue.pop() {
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
                        cur_dist: distances[&target].0,
                    })
                }
            }

            visited.insert(node);
        }

        None
    }
}

#[derive(Debug)]
pub struct UnGraph<T, W> {
    nodes: HashSet<T>,
    edges: HashMap<T, HashMap<T, W>>,

    empty: HashMap<T, W>,
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

    fn nodes(&self) -> NodeIter<'_, T> {
        NodeIter {
            nodes_iter: self.nodes.iter(),
        }
    }

    fn edges(&self, n: &T) -> EdgeIter<'_, T, W> {
        let edges = self.edges.get(n);

        if let Some(edges) = edges {
            EdgeIter {
                edge_iter: edges.iter(),
            }
        } else {
            EdgeIter {
                edge_iter: self.empty.iter(),
            }
        }
    }
}

fn copy_tuple<T, W>(x: (&T, &W)) -> (T, W)
where
    T: Clone + Copy,
    W: Clone + Copy,
{
    (*x.0, *x.1)
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

#[derive(Debug)]
pub struct DiGraph<T, W> {
    nodes: HashSet<T>,
    edges: HashMap<T, HashMap<T, W>>,

    empty: HashMap<T, W>,
}

impl<T, W> Default for DiGraph<T, W> {
    fn default() -> Self {
        DiGraph {
            nodes: HashSet::new(),
            edges: HashMap::new(),
            empty: HashMap::new(),
        }
    }
}

impl<T, W> Graph<T, W> for DiGraph<T, W>
where
    T: Clone + Copy + Eq + Hash + PartialEq,
    W: Clone + Copy,
{
    fn add_edge(&mut self, from: T, to: T, weight: W) {
        self.edges.entry(from).or_default().insert(to, weight);
    }

    fn add_node(&mut self, node: T) -> bool {
        self.nodes.insert(node)
    }

    fn remove_edge(&mut self, from: T, to: T) -> Result<()> {
        self.edges
            .get_mut(&from)
            .ok_or(anyhow!("Node not found"))?
            .remove(&to);

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

    fn nodes(&self) -> NodeIter<'_, T> {
        NodeIter {
            nodes_iter: self.nodes.iter(),
        }
    }

    fn edges(&self, n: &T) -> EdgeIter<T, W> {
        let edges = self.edges.get(n);

        if let Some(edges) = edges {
            EdgeIter {
                edge_iter: edges.iter(),
            }
        } else {
            EdgeIter {
                edge_iter: self.empty.iter(),
            }
        }
    }
}

impl<T, W> ArithmeticallyWeightedGraph<T, W> for DiGraph<T, W>
where
    T: Clone + Copy + Eq + Hash + PartialEq,
    W: Clone + Copy + std::ops::Add<Output = W> + PartialOrd + Ord + Default,
{
}

#[cfg(test)]
mod tests {
    use super::*;

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

        assert_eq!(graph.bfs(1, |x| *x == 3), Some((3, 2)));

        graph.add_edge(1, 3, 15);

        assert_eq!(graph.bfs(1, |x| *x == 3), Some((3, 1)));
    }
}
