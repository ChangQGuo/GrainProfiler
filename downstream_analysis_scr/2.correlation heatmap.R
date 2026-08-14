#=================================================
# Publication-level Correlation Matrix
# PCA + Morphological Traits
# Hexbin Version (Final Optimized)
#=================================================

library(GGally)
library(ggplot2)
library(dplyr)
library(hexbin)
library(grid)

#=================================================
# 工作目录
#=================================================

work_dir <- "C:/Users/HP/Desktop/Academic Presentation/seed_project/a_aguo_test_new/结果分析-脚本/Correlation Analysis"

input_file <- file.path(
  work_dir,
  "PCA_scores_PC1_PC5.txt"
)

#=================================================
# 读取数据
#=================================================

df <- read.table(
  input_file,
  header = TRUE,
  sep = "\t",
  check.names = FALSE
)

rownames(df) <- df$sample_id

df <- df %>%
  dplyr::select(-sample_id)

#=================================================
# 变量名称
#=================================================

colnames(df) <- c(
  "PC1","PC2","PC3","PC4","PC5",
  "Length","Max Width","Area",
  "Perimeter","Circularity","Aspect Ratio"
)

#=================================================
# 左下角：相关性（数字 + 显著性）
#=================================================

my_cor <- function(data, mapping, ...) {
  
  x <- GGally::eval_data_col(data, mapping$x)
  y <- GGally::eval_data_col(data, mapping$y)
  
  r <- cor(x, y, method = "pearson", use = "complete.obs")
  p <- cor.test(x, y)$p.value
  
  stars <- ifelse(p < 0.001, "***",
                  ifelse(p < 0.01, "**",
                         ifelse(p < 0.05, "*", "")))
  
  label <- paste0(
    sprintf("%.2f", r),
    "\n",
    stars
  )
  
  fill_col <- colorRampPalette(
    c("#FFFFFF","#EAF4FB","#A9D6E5",
      "#5FA8D3","#2C7FB8","#084081")
  )(100)
  
  idx <- round(abs(r) * 99) + 1
  bg <- fill_col[idx]
  
  ggplot() +
    annotate(
      "rect",
      xmin = 0, xmax = 1,
      ymin = 0, ymax = 1,
      fill = bg,
      colour = "white",
      linewidth = 0.4
    ) +
    annotate(
      "text",
      x = 0.5, y = 0.5,
      label = label,
      size = 8,
      fontface = "bold",
      colour = "red"
    ) +
    theme_void(base_family = "Arial")
}

#=================================================
# 右上角：Hexbin + 回归
#=================================================

my_hex <- function(data, mapping, ...) {
  
  ggplot(data = data, mapping = mapping) +
    geom_hex(bins = 20) +
    scale_fill_gradient(low = "#EAF4FB", high = "#084081") +
    geom_smooth(
      method = "lm",
      se = TRUE,
      linewidth = 0.6,
      colour = "#08306B",
      fill = "#9ECAE1",
      alpha = 0.2
    ) +
    theme_bw(base_family = "Arial") +
    theme(
      legend.position = "none",
      panel.grid = element_blank(),
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      axis.title = element_blank()
    )
}

#=================================================
# 对角线 density
#=================================================

my_density <- function(data, mapping, ...) {
  
  x <- GGally::eval_data_col(data, mapping$x)
  
  ggplot(data = data, mapping = mapping) +
    geom_density(
      fill = "#9ECAE1",
      colour = "#08519C",
      linewidth = 0.9,
      alpha = 0.75
    ) +
    geom_vline(
      xintercept = mean(x, na.rm = TRUE),
      colour = "#08306B",
      linetype = 2,
      linewidth = 0.7
    ) +
    theme_bw(base_family = "Arial") +
    theme(
      panel.grid = element_blank(),
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      axis.title = element_blank()
    )
}

#=================================================
# 主图
#=================================================

p <- ggpairs(
  df,
  
  upper = list(continuous = my_hex),
  lower = list(continuous = my_cor),
  diag  = list(continuous = my_density),
  
  switch = "both",
  progress = FALSE
)

#=================================================
# 全局优化（关键：防止挤压 + 放大变量名）
#=================================================

p <- p +
  theme_bw(base_family = "Arial") +
  theme(
    
    #=========================
    # 左 + 下变量名（放大）
    #=========================
    strip.text.x = element_text(
      size = 18,
      face = "bold",
      margin = margin(t = 6, b = 6)
    ),
    
    strip.text.y = element_text(
      size = 18,
      face = "bold",
      margin = margin(l = 6, r = 6)
    ),
    
    #=========================
    # 防止 label 和图挤压
    #=========================
    axis.text.x = element_text(
      size = 10,
      margin = margin(t = 4)
    ),
    
    axis.text.y = element_text(
      size = 10,
      margin = margin(r = 4)
    ),
    
    #=========================
    # 面板间距（关键防重叠）
    #=========================
    panel.spacing = unit(1.2, "lines"),
    
    strip.background = element_blank()
  )

#=================================================
# 保存图
#=================================================

ggsave(
  filename = file.path(
    work_dir,
    "PCA_Morphology_Hexbin_Correlation.png"
  ),
  plot = p,
  width = 20,
  height = 20,
  dpi = 600,
  bg = "white"
)

#=================================================
# 输出相关矩阵
#=================================================

cor_mat <- cor(df, method = "pearson", use = "complete.obs")

write.csv(
  round(cor_mat, 4),
  file.path(work_dir, "PCA_Correlation_Matrix.csv")
)

cat("\nFinished!\nOutput:", work_dir, "\n")