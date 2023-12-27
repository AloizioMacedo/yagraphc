use anyhow::Result;

use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::hash::Hash;

pub struct BfsIter<'a, T, W> {
    pub queue: VecDeque<(T, usize)>,
    pub visited: HashSet<T>,
    pub graph: &'a dyn Graph<T, W>,
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
    pub queue: VecDeque<(T, usize)>,
    pub visited: HashSet<T>,
    pub graph: &'a dyn Graph<T, W>,
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

pub struct NodeIter<'a, T> {
    pub nodes_iter: std::collections::hash_set::Iter<'a, T>,
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
    pub edge_iter: std::collections::hash_map::Iter<'a, T, W>,
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
    pub edge_iter: core::slice::Iter<'a, (T, W)>,
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

pub trait Graph<T, W>
where
    T: Clone + Copy + Eq + Hash + PartialEq,
    W: Clone + Copy,
{
    fn add_edge(&mut self, from: T, to: T, weight: W);

    fn add_node(&mut self, node: T) -> bool;

    fn remove_edge(&mut self, from: T, to: T) -> Result<()>;

    fn remove_node(&mut self, node: T) -> Result<()>;

    fn edges(&self, n: &T) -> EdgeIterType<T, W>;

    fn nodes(&self) -> NodeIter<T>;

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
    pub cur_dist: W,
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
                        cur_dist: distance,
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
                        cur_dist: distances[&target].0,
                    })
                }
            }

            visited.insert(node);
        }

        None
    }
}
