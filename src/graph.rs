use anyhow::{anyhow, Result};

use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::Hash;

#[derive(Debug)]
pub(crate) struct Graph<T, W> {
    nodes: HashSet<T>,
    edges: HashMap<T, HashMap<T, W>>,

    empty: HashMap<T, W>,
}

impl<T, W> Graph<T, W>
where
    T: Clone + Copy + Hash + Eq + PartialEq,
    W: Clone + Copy,
{
    pub fn add_edge(&mut self, from: T, to: T, weight: W) {
        self.edges.entry(from).or_default().insert(to, weight);
        self.edges.entry(to).or_default().insert(from, weight);

        self.nodes.insert(from);
        self.nodes.insert(to);
    }

    pub fn add_node(&mut self, node: T) -> bool {
        self.nodes.insert(node)
    }

    pub fn remove_edge(&mut self, from: T, to: T) -> Result<()> {
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

    pub fn remove_node(&mut self, node: T) -> Result<()> {
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

    pub fn nodes(&self) -> impl Iterator<Item = &T> + '_ {
        self.nodes.iter()
    }

    pub fn edges(&self, n: &T) -> impl Iterator<Item = (&T, &W)> + '_ {
        let edges = self.edges.get(n);

        if let Some(edges) = edges {
            edges.iter()
        } else {
            self.empty.iter()
        }
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

impl<T, W> Graph<T, W>
where
    T: Clone + Copy + Hash + Eq + PartialEq,
    W: Clone + Copy + std::ops::Add<Output = W> + PartialOrd + Ord + Default,
{
    pub fn dijkstra(&self, from: T, to: T) -> Option<W> {
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

            for (&target, &weight) in self.edges(&node) {
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

    pub fn dijkstra_with_path(&self, from: T, to: T) -> Option<(Vec<T>, W)> {
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

            for (&target, &weight) in self.edges(&node) {
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

impl<T, W> Default for Graph<T, W> {
    fn default() -> Self {
        Graph {
            nodes: HashSet::new(),
            edges: HashMap::new(),
            empty: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dijkstra() {
        let mut graph = Graph::default();

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
}
