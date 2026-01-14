build_adj_matrix <- function(N_items, N_comparisons, idx1, idx2) {
  adj <- matrix(0, nrow=N_items, ncol=N_items)

  # Compute directed edges
  for (n in 1:N_comparisons) {
    adj[idx1[n], idx2[n]] <- adj[idx1[n], idx2[n]] + 1
  }

  # Compute symmetric, undirected edges
  for (p1 in 1:(N_items - 1)) {
    for (p2 in (p1 + 1):N_items) {
      N1 <- adj[p1, p2]
      N2 <- adj[p2, p1]
      adj[p1, p2] <- N1 + N2
      adj[p2, p1] <- N1 + N2
    }
  }

  (adj)
}

breadth_first_search <- function(n, adj, local_nodes=c()) {
  local_nodes <- c(local_nodes, n)
  neighbors <- which(adj[n,] > 0)
  for (nn in neighbors) {
    if (!(nn %in% local_nodes))
      local_nodes <- union(local_nodes,
                           Recall(nn, adj, local_nodes))
  }
  (local_nodes)
}

compute_connected_components <- function(adj) {
  if (!is.matrix(adj)) {
    stop("Input adj is not a matrix.")
  }
  if (nrow(adj) != ncol(adj)) {
    stop("Input adj is not a square matrix.")
  }

  all_nodes <- 1:nrow(adj)
  remaining_nodes <- all_nodes
  components <- list()

  for (n in all_nodes) {
    if (n %in% remaining_nodes) {
      subgraph <- breadth_first_search(n, adj)
      remaining_nodes <- setdiff(remaining_nodes, subgraph)
      components[[length(components) + 1]] <- subgraph
    }
  }

  (components)
}