mod graph;

fn main() {
    let mut graph = graph::Graph::default();

    graph.add_edge(1, 2, 3);
    graph.add_edge(2, 3, 10);

    println!("{:?}", graph);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {}
}
