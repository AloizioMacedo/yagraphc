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

pub struct PostOrderDfsIter<'a, T, W> {
    pub stack: Vec<(T, usize)>,
    pub visited: HashSet<T>,
    pub graph: &'a dyn Graph<T, W>,
}

impl<'a, T, W> PostOrderDfsIter<'a, T, W>
where
    T: Clone + Copy + Hash + PartialEq + Eq,
    W: Clone + Copy,
{
    pub fn new(graph: &'a dyn Graph<T, W>, from: T) -> Self {
        let mut stack1 = Vec::new();
        let mut stack2 = Vec::new();

        let mut visited = HashSet::new();
        stack1.push((from, 0));

        while let Some((node, depth)) = stack1.pop() {
            if visited.contains(&node) {
                continue;
            }

            stack2.push((node, depth));
            visited.insert(node);

            for (next, _) in graph.edges(&node) {
                stack1.push((next, depth + 1));
            }
        }

        Self {
            stack: stack2,
            visited,
            graph,
        }
    }
}

impl<'a, T, W> Iterator for PostOrderDfsIter<'a, T, W>
where
    T: Clone + Copy + Hash + PartialEq + Eq,
    W: Clone + Copy,
{
    type Item = (T, usize);

    fn next(&mut self) -> Option<Self::Item> {
        self.stack.pop()
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

    fn in_edges(&self, n: &T) -> EdgeIterType<T, W>;

    fn has_edge(&self, from: T, to: T) -> bool;

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

    fn dfs_post_order(&self, from: T) -> PostOrderDfsIter<T, W>
    where
        Self: Sized,
    {
        PostOrderDfsIter::new(self, from)
    }

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

    fn a_star<G>(&self, from: T, to: T, heuristic: G) -> Option<W>
    where
        Self: Graph<T, W>,
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

    fn a_star_with_path<G>(&self, from: T, to: T, heuristic: G) -> Option<(Vec<T>, W)>
    where
        Self: Graph<T, W>,
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
}
