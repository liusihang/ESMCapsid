#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(patchwork)
  library(readr)
  library(scales)
  library(svglite)
})

this_file <- function() {
  frame_files <- Filter(Negate(is.null), lapply(sys.frames(), function(x) x$ofile))
  if (length(frame_files) > 0) {
    return(normalizePath(tail(frame_files, 1)[[1]]))
  }
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    candidate <- sub("^--file=", "", file_arg[[1]])
    if (candidate != "-" && file.exists(candidate)) {
      return(normalizePath(candidate))
    }
  }
  stop("Unable to determine current script path")
}

FIGURE_ROOT_ENV <- Sys.getenv("FIG6_ROOT", unset = "")
if (nzchar(FIGURE_ROOT_ENV)) {
  FIGURE_ROOT <- normalizePath(FIGURE_ROOT_ENV)
  SCRIPT_DIR <- file.path(FIGURE_ROOT, "scripts")
  SCRIPT_PATH <- file.path(SCRIPT_DIR, "build_panel.R")
} else {
  SCRIPT_PATH <- this_file()
  SCRIPT_DIR <- dirname(SCRIPT_PATH)
  FIGURE_ROOT <- dirname(SCRIPT_DIR)
}
PANEL_A_CSV <- file.path(FIGURE_ROOT, "data", "panel_a_fold_summary.csv")
PANEL_B_CSV <- file.path(FIGURE_ROOT, "data", "panel_b_ecosystem_enrichment.csv")
PANEL_C_ASSOCIATION_CSV <- file.path(
  FIGURE_ROOT,
  "data",
  "panel_c_taxonomy_associations.csv"
)
SUB_COUNTS_CSV <- file.path(
  FIGURE_ROOT,
  "data",
  "panel_c_fold_by_sub_ecosystem_counts.csv"
)
PANEL_OUTPUT_DIR <- file.path(FIGURE_ROOT, "plots")

BASE_FAMILY <- "Helvetica"
TEXT_COLOR <- "#1F2A36"
MUTED <- "#5E6C7B"
GRID <- "#E5E9EF"
BEIGE_LOW <- "#F2E6D2"
ORANGE_HIGH <- "#C96C2F"
BURD <- c(
  "#2166AC",
  "#4393C3",
  "#92C5DE",
  "#D1E5F0",
  "#F7F7F7",
  "#FDDBC7",
  "#F4A582",
  "#D6604D",
  "#B2182B"
)

ENV_ORDER <- c(
  "General Environmental",
  "Host Associated",
  "Extreme Environments",
  "Engineered Systems",
  "Contaminated and Industrial Environments",
  "Unclassified",
  "Mixed And Lab"
)
PANEL_C_TAXONOMY_LEVEL <- "Phylum"

format_count <- function(x) {
  ifelse(
    x >= 1e6, sprintf("%.2fM", x / 1e6),
    ifelse(
      x >= 1e5, sprintf("%.0fk", x / 1e3),
      ifelse(
        x >= 1e4, sprintf("%.1fk", x / 1e3),
        ifelse(x >= 1e3, sprintf("%.1fk", x / 1e3), as.character(round(x)))
      )
    )
  )
}

get_fold_order <- function() {
  read_csv(PANEL_A_CSV, show_col_types = FALSE) |>
    filter(!manuscript_group %in% c("(no manuscript_group)", "Delete_marked", "Sample-SJR")) |>
    mutate(expanded_member_count = as.numeric(expanded_member_count)) |>
    arrange(desc(expanded_member_count)) |>
    pull(manuscript_group) |>
    unique()
}

pretty_env_full <- function(x) {
  recode(
    x,
    "Animals" = "Animals",
    "Aquatic Freshwater" = "Freshwater",
    "Aquatic Marine" = "Marine",
    "Aquatic Special" = "Aquatic special",
    "Atmospheric" = "Atmospheric",
    "Bioreactors" = "Bioreactors",
    "Built Environment" = "Built env.",
    "Chemical Remediation Targets" = "Chem. remediation",
    "Cleanrooms" = "Cleanrooms",
    "Controlled Environment" = "Controlled env.",
    "Food Production" = "Food prod.",
    "Fungi And Algae" = "Fungi/algae",
    "Halophilic And Alkaliphilic" = "Halo/alkali",
    "Human" = "Human",
    "Hydrocarbon Contamination Oil" = "Hydrocarbon",
    "Industrial Hydrocarbon Processing" = "Ind. hydrocarbon",
    "Industrial Processing" = "Ind. processing",
    "Industrial Wastewater" = "Ind. wastewater",
    "Lab Enrichment" = "Lab enrich.",
    "Landfill" = "Landfill",
    "Microbial Hosts" = "Microbial host",
    "Modeled" = "Modeled",
    "Plants" = "Plants",
    "Psychrophilic Low Temperature" = "Psychro.",
    "Solid Waste Management" = "Solid waste",
    "Subsurface And Geologic" = "Subsurface",
    "Tailings Pond" = "Tailings",
    "Terrestrial Soil" = "Soil",
    "Terrestrial Vegetation" = "Vegetation",
    "Thermophilic High Temperature" = "Thermo.",
    "Unclassified" = "Unclassified",
    "Unknown Mixed" = "Unknown mix",
    "Wastewater Treatment" = "Wastewater",
    .default = x
  )
}

panel_theme <- function() {
  theme_minimal(base_family = BASE_FAMILY, base_size = 8) +
    theme(
      text = element_text(colour = TEXT_COLOR),
      axis.title = element_text(size = 9, colour = TEXT_COLOR),
      axis.text = element_text(size = 8, colour = MUTED),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_blank(),
      plot.margin = margin(8, 10, 8, 10)
    )
}

load_panel_a_data <- function() {
  fold_order <- get_fold_order()
  read_csv(PANEL_A_CSV, show_col_types = FALSE) |>
    filter(!manuscript_group %in% c("(no manuscript_group)", "Delete_marked", "Sample-SJR")) |>
    mutate(
      across(c(representative_count, expanded_member_count, expanded_hmm_hit_count, expanded_hmm_hit_rate, redundancy_fold), as.numeric)
    ) |>
    arrange(desc(expanded_member_count)) |>
    mutate(
      manuscript_group = factor(manuscript_group, levels = fold_order),
      log_members = log10(expanded_member_count),
      log_hits = log10(pmax(expanded_hmm_hit_count, 1)),
      y = seq_len(n())
    )
}

load_panel_b_data <- function() {
  fold_order <- get_fold_order()
  dat <- read_csv(PANEL_B_CSV, show_col_types = FALSE) |>
    mutate(
      abundance = as.numeric(normalized_abundance_within_group),
      log2_enrichment = suppressWarnings(as.numeric(log2_enrichment)),
      q_value = as.numeric(q_value)
    )
  floor_val <- min(dat$log2_enrichment, na.rm = TRUE) - 0.5
  dat |>
    mutate(
      fold = factor(manuscript_group, levels = fold_order),
      env = factor(main_ecosystem, levels = rev(ENV_ORDER)),
      x = match(fold, fold_order),
      y = match(main_ecosystem, rev(ENV_ORDER)),
      log2_plot = ifelse(is.na(log2_enrichment), floor_val, log2_enrichment),
      sig = !is.na(log2_enrichment) & q_value < 0.01 & abs(log2_enrichment) >= 0.5
    )
}

load_panel_c_data <- function() {
  fold_order <- get_fold_order()
  fold_wide <- read_csv(SUB_COUNTS_CSV, show_col_types = FALSE)
  sub_ecosystems <- setdiff(colnames(fold_wide), "manuscript_group")

  fold_long <- bind_rows(lapply(seq_len(nrow(fold_wide)), function(i) {
    row <- fold_wide[i, ]
    tibble(
      label = row$manuscript_group[[1]],
      sub_ecosystem = sub_ecosystems,
      count = as.numeric(row[1, sub_ecosystems]),
      source_block = "Fold/group",
      source_level = "manuscript_group"
    )
  })) |>
    filter(label %in% fold_order) |>
    group_by(sub_ecosystem) |>
    mutate(
      ecosystem_total = sum(count, na.rm = TRUE),
      pct_composition = if_else(ecosystem_total > 0, count / ecosystem_total * 100, 0)
    ) |>
    select(-ecosystem_total) |>
    ungroup()

  assoc_long <- read_csv(PANEL_C_ASSOCIATION_CSV, show_col_types = FALSE) |>
    filter(Ecosystem_Type == "sub_ecosystem") |>
    transmute(
      source_level = Taxonomic_Level,
      label = recode(
        Taxon_Name,
        "Unclassified_Unknown" = "Unclassified",
        "ssRNA(+)" = "ssRNA",
        .default = Taxon_Name
      ),
      sub_ecosystem = Ecosystem_Name,
      count = as.numeric(Count),
      pct_composition = as.numeric(Pct_Composition_of_Ecosystem)
    )

  taxonomy_long <- assoc_long |>
    filter(source_level == PANEL_C_TAXONOMY_LEVEL)
  taxonomy_order <- taxonomy_long |>
    group_by(label) |>
    summarise(total_count = sum(count, na.rm = TRUE), .groups = "drop") |>
    arrange(desc(total_count), label) |>
    pull(label)

  genome_long <- assoc_long |>
    filter(source_level == "GenomeLabel")
  genome_order <- genome_long |>
    group_by(label) |>
    summarise(total_count = sum(count, na.rm = TRUE), .groups = "drop") |>
    arrange(desc(total_count), label) |>
    pull(label)

  complete_block <- function(data, labels, block_name) {
    expand.grid(label = labels, sub_ecosystem = sub_ecosystems, stringsAsFactors = FALSE) |>
      as_tibble() |>
      left_join(data, by = c("label", "sub_ecosystem")) |>
      mutate(
        count = ifelse(is.na(count), 0, count),
        pct_composition = ifelse(is.na(pct_composition), 0, pct_composition),
        source_block = block_name
      )
  }

  taxonomy_full <- complete_block(taxonomy_long, taxonomy_order, "Taxonomy (Phylum)")
  genome_full <- complete_block(genome_long, genome_order, "Genome type")

  fold_full <- fold_long |>
    mutate(
      label = factor(label, levels = fold_order),
      source_block = "Fold/group"
    )

  blocks <- list(
    "Fold/group" = fold_full |> mutate(label = factor(as.character(label), levels = fold_order)),
    "Taxonomy (Phylum)" = taxonomy_full |> mutate(label = factor(label, levels = taxonomy_order)),
    "Genome type" = genome_full |> mutate(label = factor(label, levels = genome_order))
  )

  max_pct <- max(
    c(
      fold_full$pct_composition,
      taxonomy_full$pct_composition,
      genome_full$pct_composition
    ),
    na.rm = TRUE
  )
  max_count <- max(
    c(
      fold_full$count,
      taxonomy_full$count,
      genome_full$count
    ),
    na.rm = TRUE
  )

  list(
    blocks = blocks,
    sub_ecosystems = sub_ecosystems,
    max_pct = max_pct,
    max_count = max_count
  )
}

build_panel_a <- function(dat) {
  positive_vals <- c(dat$expanded_member_count[dat$expanded_member_count > 0], dat$expanded_hmm_hit_count[dat$expanded_hmm_hit_count > 0])
  x_min_count <- 10 ^ floor(log10(min(positive_vals)))
  x_break_counts <- c(1e2, 1e3, 1e4, 1e5, 1e6)
  x_break_counts <- x_break_counts[x_break_counts >= x_min_count & x_break_counts <= max(dat$expanded_member_count) * 1.2]
  x_breaks <- log10(x_break_counts)
  x_min <- log10(x_min_count)

  ggplot(dat) +
    geom_vline(xintercept = x_breaks, colour = GRID, linewidth = 0.5) +
    geom_rect(
      aes(
        xmin = x_min, xmax = log_members,
        ymin = y - 0.31, ymax = y + 0.31,
        fill = redundancy_fold
      ),
      colour = NA,
      alpha = 0.35
    ) +
    geom_rect(
      aes(
        xmin = x_min, xmax = log_hits,
        ymin = y - 0.31, ymax = y + 0.31,
        fill = redundancy_fold
      ),
      colour = "#7A4B24",
      linewidth = 0.15,
      alpha = 0.95,
      inherit.aes = FALSE,
      data = dat
    ) +
    scale_fill_gradient(
      low = BEIGE_LOW,
      high = ORANGE_HIGH,
      name = "Redundancy",
      guide = guide_colorbar(
        title.position = "top",
        title.hjust = 0.5,
        barwidth = unit(4.8, "cm"),
        barheight = unit(0.3, "cm")
      )
    ) +
    scale_x_continuous(
      breaks = x_breaks,
      labels = scales::label_math(10^.x)(log10(x_break_counts)),
      limits = c(x_min, max(dat$log_members) + 0.22),
      expand = expansion(mult = c(0.01, 0.01))
    ) +
    scale_y_reverse(
      breaks = dat$y,
      labels = dat$manuscript_group,
      expand = expansion(mult = c(0.02, 0.22))
    ) +
    labs(
      x = "Expanded member count (log10 scale)",
      y = NULL
    ) +
    panel_theme() +
    theme(
      axis.text.y = element_text(size = 8, colour = TEXT_COLOR),
      panel.grid.major = element_blank(),
      axis.line.x = element_line(colour = "#B9C2CC", linewidth = 0.4),
      legend.position = c(0.66, 0.05),
      legend.direction = "horizontal",
      legend.justification = c(0.5, 0),
      legend.title = element_text(size = 8, colour = MUTED),
      legend.text = element_text(size = 8, colour = MUTED),
      plot.margin = margin(8, 40, 18, 12)
    ) +
    coord_cartesian(clip = "off")
}

build_panel_b_nature <- function(dat) {
  dat_sig <- dat |> filter(sig == TRUE)

  nature_colors <- c(
    "HK97-like"    = "#4DBBD5",
    "NCLDV-like"   = "#E64B35",
    "picorna-like" = "#00A087",
    "BTV-like"     = "#3C5488",
    "micro-like"   = "#F39B7F",
    "levi-like"    = "#7E6148",
    "ino-like"     = "#8491B4",
    "Geminiviridae-like" = "#7E57C2",
    "Circoviridae-like" = "#B8860B"
  )
  
  ggplot(dat_sig, aes(x = log2_plot, y = env)) +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "#A0AAB5", linewidth = 0.6) +

    geom_point(
      aes(size = abundance, fill = fold), 
      shape = 21, 
      alpha = 0.9, 
      colour = "white",
      stroke = 0.5,
      position = position_jitter(height = 0.15, seed = 42)
    ) +

    scale_size_continuous(
      name = "Relative Abundance",
      range = c(2.5, 11),
      breaks = c(0.1, 0.3, 0.6),
      labels = c("10%", "30%", "60%")
    ) +

    scale_fill_manual(values = nature_colors, name = "Virus Fold") + 

    scale_y_discrete(drop = FALSE, labels = rev(ENV_ORDER)) +
    labs(x = "Log2 Enrichment (Significant only)", y = NULL) +

    panel_theme() +
    theme(
      panel.border = element_rect(colour = "#1F2A36", fill = NA, linewidth = 0.8),
      axis.line = element_blank(), 
      axis.ticks = element_line(colour = "#1F2A36", linewidth = 0.5), 
      axis.ticks.length = unit(0.15, "cm"),
      panel.grid.major.y = element_line(colour = "#EBEFF5", linetype = "solid", linewidth = 0.5),
      panel.grid.major.x = element_blank(),
      legend.position = "right",
      legend.spacing.y = unit(0.2, "cm"),
      legend.key = element_blank(),
      legend.key.size = unit(0.8, "cm")
    ) +
    guides(
      size = guide_legend(override.aes = list(fill = "#8C9BB0", colour = "white", stroke = 0.5)),
      fill = guide_legend(override.aes = list(size = 5))
    )
}

build_panel_c_block <- function(dat, sub_ecosystems, global_max_count, block_title, show_x = FALSE) {
  
  dat <- dat |>
    mutate(
      x = match(sub_ecosystem, sub_ecosystems),
      y = rev(match(as.character(label), levels(label))),
      label_chr = as.character(label)
    )
  
  label_df <- dat |> distinct(label_chr, y) |> arrange(desc(y))
  point_df <- dat |> filter(count > 0)
  
  p <- ggplot(dat, aes(x = x, y = y)) +
    geom_tile(fill = NA, colour = "#EAEEF2", linewidth = 0.3) +

    geom_point(
      data = point_df,
      aes(size = count, fill = pct_composition),
      shape = 21,
      colour = "#B0B8C1", 
      stroke = 0.25,
      alpha = 0.90 
    ) +
    
    scale_x_continuous(
      breaks = seq_along(sub_ecosystems),
      labels = pretty_env_full(sub_ecosystems),
      expand = c(0, 0)
    ) +
    scale_y_continuous(
      breaks = label_df$y,
      labels = label_df$label_chr,
      expand = c(0, 0)
    ) +
    
    scale_size_continuous(
      name = "Absolute count",
      trans = "log10",
      range = c(0.8, 5.5), 
      limits = c(1, global_max_count),
      breaks = c(1e2, 1e3, 1e4, 1e5),
      labels = format_count
    ) +

    scale_fill_gradientn(
      colors = BURD,
      limits = c(0, 100), 
      trans = "sqrt",
      name = "Ecosystem composition (%)",
      breaks = c(1, 10, 30, 60, 100),
      guide = guide_colorbar(
        title.position = "top",
        title.hjust = 0.5,
        barwidth = unit(4.8, "cm"),
        barheight = unit(0.25, "cm"),
        order = 1,
        frame.colour = "#D0D6E0", 
        ticks.colour = "white"    
      )
    ) +
    
    labs(
      title = block_title,
      x = if (show_x) "Sub-ecosystems" else NULL,
      y = NULL
    ) +
    panel_theme() +
    theme(
      plot.title = element_text(size = 9, face = "bold", hjust = 0, colour = TEXT_COLOR),
      axis.text.y = element_text(size = 7.5, colour = TEXT_COLOR),
      axis.text.x = if (show_x) element_text(angle = 45, hjust = 1, vjust = 1, size = 7.5, colour = TEXT_COLOR) else element_blank(),
      axis.ticks.x = element_blank(),
      panel.grid = element_blank(),
      legend.position = if (show_x) "bottom" else "none", 
      legend.box = "horizontal",
      legend.title = element_text(size = 8, colour = MUTED),
      legend.text = element_text(size = 8, colour = MUTED),
      plot.margin = margin(4, 8, if (show_x) 10 else 2, 8)
    )
  
  if (!show_x) p <- p + theme(axis.title.x = element_blank())
  p
}

build_panel_c <- function(c_obj) {
  blocks <- c_obj$blocks
  sub_ecosystems <- c_obj$sub_ecosystems

  p1 <- build_panel_c_block(
    blocks[["Fold/group"]],
    sub_ecosystems = sub_ecosystems,
    global_max_count = c_obj$max_count,
    block_title = "Fold/group",
    show_x = FALSE
  )
  p2 <- build_panel_c_block(
    blocks[["Taxonomy (Phylum)"]],
    sub_ecosystems = sub_ecosystems,
    global_max_count = c_obj$max_count,
    block_title = "Taxonomy (Phylum)",
    show_x = FALSE
  )
  p3 <- build_panel_c_block(
    blocks[["Genome type"]],
    sub_ecosystems = sub_ecosystems,
    global_max_count = c_obj$max_count,
    block_title = "Genome type",
    show_x = TRUE
  )

  p1 / p2 / p3 +
    plot_layout(
      heights = c(
        max(4, nlevels(blocks[["Fold/group"]]$label)),
        max(5, nlevels(blocks[["Taxonomy (Phylum)"]]$label)),
        max(4, nlevels(blocks[["Genome type"]]$label))
      ),
      guides = "collect"
    )
}

parse_panel <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  idx <- match("--panel", args)
  panel <- if (!is.na(idx) && idx < length(args)) args[[idx + 1]] else ""
  panel <- toupper(panel)
  if (!panel %in% c("A", "B", "C")) {
    stop("Usage: Rscript build_panel.R --panel A|B|C")
  }
  panel
}

save_panel <- function(panel) {
  dir.create(PANEL_OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
  plot_obj <- switch(
    panel,
    A = build_panel_a(load_panel_a_data()),
    B = build_panel_b_nature(load_panel_b_data()),
    C = build_panel_c(load_panel_c_data())
  )
  dimensions <- switch(
    panel,
    A = c(width = 8.0, height = 7.0),
    B = c(width = 8.0, height = 7.0),
    C = c(width = 12.0, height = 14.0)
  )
  prefix <- file.path(PANEL_OUTPUT_DIR, paste0("panel_", tolower(panel), "_reproduced"))
  ggsave(paste0(prefix, ".png"), plot_obj, width = dimensions[["width"]], height = dimensions[["height"]], dpi = 600, bg = "white")
  ggsave(paste0(prefix, ".pdf"), plot_obj, width = dimensions[["width"]], height = dimensions[["height"]], bg = "white", device = grDevices::pdf)
  ggsave(paste0(prefix, ".svg"), plot_obj, width = dimensions[["width"]], height = dimensions[["height"]], bg = "white", device = svglite)
  cat(paste0(prefix, ".png"), "\n")
  cat(paste0(prefix, ".pdf"), "\n")
  cat(paste0(prefix, ".svg"), "\n")
}

if (sys.nframe() == 0) {
  save_panel(parse_panel())
}
