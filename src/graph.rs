use anyhow::{anyhow, Result};

use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::Hash;

#[derive(Debug)]
pub(crate) struct Graph<T, W> {
    nodes: HashSet<T>,
    edges: HashMap<T, HashMap<T, W>>,
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

    pub fn edges(&self, n: &T) -> Option<impl Iterator<Item = (T, W)> + '_> {
        self.edges
            .get(n)
            .map(|edges| edges.iter().map(|(k, v)| (*k, *v)))
    }
}

impl<T, W> Default for Graph<T, W> {
    fn default() -> Self {
        Graph {
            nodes: HashSet::new(),
            edges: HashMap::new(),
        }
    }
}
