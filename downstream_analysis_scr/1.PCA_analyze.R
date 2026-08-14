library(ggplot2)
library(patchwork)
library(ggrepel)
library(dplyr)

#====================================================
# 输出目录
#====================================================

output_dir <- "C:/Users/HP/Desktop/Academic Presentation/seed_project/a_aguo_test_new/结果分析-脚本/PCA_result"

if(!dir.exists(output_dir)){
  dir.create(output_dir, recursive = TRUE)
}

#====================================================
# 读取数据
#====================================================

data <- read.table(
  "C:/Users/HP/Desktop/Academic Presentation/seed_project/a_aguo_test_new/a_HN_result/rep_width_profiles.txt",
  row.names = 1,
  header = FALSE,
  check.names = FALSE
)

#====================================================
# PCA
#====================================================

pca_result <- prcomp(
  data,
  center = TRUE,
  scale. = TRUE
)

pca_scores <- as.data.frame(
  pca_result$x
)

pca_scores$sample_id <- rownames(
  pca_scores
)

#====================================================
# 方差解释率
#====================================================

var_exp <- round(
  100 * pca_result$sdev^2 /
    sum(pca_result$sdev^2),
  2
)

pc1_lab <- paste0(
  "PC1 (",
  var_exp[1],
  "%)"
)

pc2_lab <- paste0(
  "PC2 (",
  var_exp[2],
  "%)"
)

variance_table <- data.frame(
  PC = paste0(
    "PC",
    seq_along(var_exp)
  ),
  Variance_Explained = var_exp,
  Cumulative =
    cumsum(var_exp)
)

#====================================================
# PCA Scores
#====================================================

pca_output <- pca_scores[,c(
  "sample_id",
  "PC1",
  "PC2",
  "PC3",
  "PC4",
  "PC5"
)]

colnames(pca_output)[1] <- "Sample"

write.table(
  pca_output,
  file.path(
    output_dir,
    "PCA_scores_PC1_PC5.txt"
  ),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

write.table(
  variance_table,
  file.path(
    output_dir,
    "PCA_variance_explained.txt"
  ),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

#====================================================
# 每个象限最远点
#====================================================

pca_scores$dist_to_origin <-
  sqrt(
    pca_scores$PC1^2 +
      pca_scores$PC2^2
  )

pca_scores$quadrant <- case_when(
  pca_scores$PC1 >= 0 &
    pca_scores$PC2 >= 0 ~ "Q1",
  
  pca_scores$PC1 < 0 &
    pca_scores$PC2 >= 0 ~ "Q2",
  
  pca_scores$PC1 < 0 &
    pca_scores$PC2 < 0 ~ "Q3",
  
  TRUE ~ "Q4"
)

extreme_points <-
  pca_scores %>%
  group_by(quadrant) %>%
  slice_max(
    order_by = dist_to_origin,
    n = 1
  ) %>%
  ungroup()

pca_scores$point_type <- ifelse(
  pca_scores$sample_id %in%
    extreme_points$sample_id,
  "extreme",
  "normal"
)

#====================================================
# PCA主图
#====================================================

p_main <-
  
  ggplot() +
  
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    colour = "grey60"
  ) +
  
  geom_hline(
    yintercept = 0,
    linetype = "dashed",
    colour = "grey60"
  ) +
  
  geom_point(
    data = subset(
      pca_scores,
      point_type=="normal"
    ),
    aes(
      PC1,
      PC2,
      colour = dist_to_origin
    ),
    alpha = 0.7,
    size = 5
  ) +
  
  geom_point(
    data = extreme_points,
    aes(
      PC1,
      PC2
    ),
    colour = "#CC3333",
    size = 5
  ) +
  
  geom_text_repel(
    data = extreme_points,
    aes(
      PC1,
      PC2,
      label = sample_id
    ),
    size = 7,
    colour = "#CC3333",
    fontface = "bold",
    box.padding = 0.5
  ) +
  
  scale_colour_gradient(
    low = "#E6F2FF",
    high = "#0066CC",
    name = "Distance"
  ) +
  
  labs(
    x = pc1_lab,
    y = pc2_lab
  ) +
  
  theme_bw() +
  
  theme(
    panel.grid = element_blank(),
    aspect.ratio = 0.8,
    
    axis.title = element_text(size = 18),
    axis.text = element_text(size = 18),
    
    legend.title = element_text(size = 18),
    legend.text = element_text(size = 18)
  )

#====================================================
# 边缘密度图
#====================================================

p_density_x <-
  
  ggplot(
    pca_scores,
    aes(PC1)
  ) +
  
  geom_density(
    fill = "grey80",
    alpha = 0.5
  ) +
  
  theme_void()

p_density_y <-
  
  ggplot(
    pca_scores,
    aes(PC2)
  ) +
  
  geom_density(
    fill = "grey80",
    alpha = 0.5
  ) +
  
  coord_flip() +
  
  theme_void()

combined_plot <-
  
  p_density_x +
  plot_spacer() +
  p_main +
  p_density_y +
  
  plot_layout(
    ncol = 2,
    nrow = 2,
    widths = c(4,1),
    heights = c(1,4)
  )

ggsave(
  file.path(
    output_dir,
    "PCA_2D_HN.png"
  ),
  combined_plot,
  width = 10,
  height = 8,
  dpi = 600
)


#====================================================
# PCA主图（Axis Extreme Points）
#====================================================

# 取最接近PC1=0的样本
near_pc1_zero <- pca_scores %>%
  mutate(abs_PC1 = abs(PC1)) %>%
  arrange(abs_PC1) %>%
  slice(1:20)

# 在这些样本中寻找PC2最大和最小
pc2_max_axis <- near_pc1_zero %>%
  slice_max(PC2, n = 1)

pc2_min_axis <- near_pc1_zero %>%
  slice_min(PC2, n = 1)

# 取最接近PC2=0的样本
near_pc2_zero <- pca_scores %>%
  mutate(abs_PC2 = abs(PC2)) %>%
  arrange(abs_PC2) %>%
  slice(1:20)

# 在这些样本中寻找PC1最大和最小
pc1_max_axis <- near_pc2_zero %>%
  slice_max(PC1, n = 1)

pc1_min_axis <- near_pc2_zero %>%
  slice_min(PC1, n = 1)

axis_extreme_points <- bind_rows(
  pc2_max_axis,
  pc2_min_axis,
  pc1_max_axis,
  pc1_min_axis
) %>%
  distinct(sample_id, .keep_all = TRUE)

#----------------------------------------------------
# PCA主图
#----------------------------------------------------

p_main2 <-
  
  ggplot() +
  
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    colour = "grey60"
  ) +
  
  geom_hline(
    yintercept = 0,
    linetype = "dashed",
    colour = "grey60"
  ) +
  
  geom_point(
    data = pca_scores,
    aes(
      PC1,
      PC2,
      colour = dist_to_origin
    ),
    alpha = 0.7,
    size = 5
  ) +
  
  geom_point(
    data = axis_extreme_points,
    aes(
      PC1,
      PC2
    ),
    colour = "#CC3333",
    size = 6
  ) +
  
  geom_text_repel(
    data = axis_extreme_points,
    aes(
      PC1,
      PC2,
      label = sample_id
    ),
    size = 7,
    colour = "#CC3333",
    fontface = "bold",
    box.padding = 0.5,
    max.overlaps = Inf
  ) +
  
  scale_colour_gradient(
    low = "#E6F2FF",
    high = "#0066CC",
    name = "Distance"
  ) +
  
  labs(
    x = pc1_lab,
    y = pc2_lab
  ) +
  
  theme_bw() +
  
  theme(
    panel.grid = element_blank(),
    aspect.ratio = 0.8,
    
    axis.title = element_text(size = 18),
    axis.text = element_text(size = 18),
    
    legend.title = element_text(size = 18),
    legend.text = element_text(size = 18)
  )

#----------------------------------------------------
# 拼图
#----------------------------------------------------

combined_plot2 <-
  
  p_density_x +
  plot_spacer() +
  p_main2 +
  p_density_y +
  
  plot_layout(
    ncol = 2,
    nrow = 2,
    widths = c(4,1),
    heights = c(1,4)
  )

#----------------------------------------------------
# 保存
#----------------------------------------------------

ggsave(
  file.path(
    output_dir,
    "PCA_2D_HN_2.png"
  ),
  combined_plot2,
  width = 10,
  height = 8,
  dpi = 600
)



#====================================================
# PCA Loadings
#====================================================

loadings_table <-
  as.data.frame(
    pca_result$rotation[,1:5]
  )

loadings_table$Relative_Position <-
  seq(
    0,
    1,
    length.out =
      nrow(loadings_table)
  )

write.csv(
  loadings_table,
  file.path(
    output_dir,
    "PCA_loadings_PC1_PC5.csv"
  ),
  row.names = FALSE
)

#====================================================
# PC1
#====================================================

p_pc1 <-
  
  ggplot(
    loadings_table,
    aes(
      Relative_Position,
      PC1
    )
  ) +
  
  geom_line(
    linewidth = 1
  ) +
  
  geom_hline(
    yintercept = 0,
    linetype = "dashed"
  ) +
  
  theme_bw() +
  
  labs(
    x = "Relative Position",
    y = "Loading",
    title = "PC1 Loading Curve"
  ) +
  
  theme(
    panel.grid =
      element_blank(),
    plot.title = element_text(
      size = 15,
      face = "bold"
    ),
    
    axis.title = element_text(
      size = 15
    ),
    
    axis.text = element_text(
      size = 15
    )
  )

ggsave(
  file.path(
    output_dir,
    "PC1_loading_curve.png"
  ),
  p_pc1,
  width = 8,
  height = 5,
  dpi = 600
)

#====================================================
# PC2
#====================================================

p_pc2 <-
  
  ggplot(
    loadings_table,
    aes(
      Relative_Position,
      PC2
    )
  ) +
  
  geom_line(
    linewidth = 1
  ) +
  
  geom_hline(
    yintercept = 0,
    linetype = "dashed"
  ) +
  
  theme_bw() +
  
  labs(
    x = "Relative Position",
    y = "Loading",
    title = "PC2 Loading Curve"
  ) +
  
  theme(
    panel.grid =
      element_blank(),
    plot.title = element_text(
      size = 15,
      face = "bold"
    ),
    
    axis.title = element_text(
      size = 15
    ),
    
    axis.text = element_text(
      size = 15
    )
  )

ggsave(
  file.path(
    output_dir,
    "PC2_loading_curve.png"
  ),
  p_pc2,
  width = 8,
  height = 5,
  dpi = 600
)

#====================================================
# PC1 vs PC2
#====================================================

loading_long <- rbind(
  
  data.frame(
    Position =
      loadings_table$Relative_Position,
    Loading =
      loadings_table$PC1,
    PC = "PC1"
  ),
  
  data.frame(
    Position =
      loadings_table$Relative_Position,
    Loading =
      loadings_table$PC2,
    PC = "PC2"
  )
)

p_compare <-
  
  ggplot(
    loading_long,
    aes(
      Position,
      Loading,
      colour = PC
    )
  ) +
  
  geom_line(
    linewidth = 1
  ) +
  
  geom_hline(
    yintercept = 0,
    linetype = "dashed"
  ) +
  
  theme_bw() +
  
  labs(
    x = "Relative Position",
    y = "Loading",
    title = "PC1 vs PC2 Loading Curve"
  ) +
  
  theme(
    panel.grid =
      element_blank(),
    plot.title = element_text(
      size = 15,
      face = "bold"
    ),
    
    axis.title = element_text(
      size = 15
    ),
    
    axis.text = element_text(
      size = 15
    ),
    
    legend.title = element_text(
      size = 15
    ),
    
    legend.text = element_text(
      size = 15
    )
  )

ggsave(
  file.path(
    output_dir,
    "PC1_PC2_loading_compare.png"
  ),
  p_compare,
  width = 8,
  height = 5,
  dpi = 600
)


cat("\n=====================================\n")
cat("PCA analysis completed!\n")
cat(output_dir,"\n")
cat("=====================================\n")