export interface ModelData {
  numeric_features: string[];
  categorical_features: string[];
  imputer_medians: number[];
  scaler_means: number[];
  scaler_stds: number[];
  onehot_categories: string[][];
  trees: {
    feature: number[];
    threshold: number[];
    children_left: number[];
    children_right: number[];
    value: number[];
  }[];
}

export interface SchoolInput {
  UGDS_log: number | null;
  ADM_RATE: number | null;
  ADM_RATE_missing: number;
  TUITIONFEE_OUT: number | null;
  NPT4: number | null;
  PCTPELL: number | null;
  PCTFLOAN: number | null;
  UGDS_WHITE: number | null;
  UGDS_BLACK: number | null;
  UGDS_HISP: number | null;
  UGDS_ASIAN: number | null;
  PCIP_reported: number;
  high_earning_share: number | null;
  CONTROL: string;
  PREDDEG: string;
  HIGHDEG: string;
  LOCALE: string;
  REGION: string;
}

function traverseTree(
  tree: ModelData["trees"][0],
  features: number[]
): number {
  let node = 0;
  while (tree.children_left[node] !== -1) {
    const feat = tree.feature[node];
    const thresh = tree.threshold[node];
    if (features[feat] <= thresh) {
      node = tree.children_left[node];
    } else {
      node = tree.children_right[node];
    }
  }
  return tree.value[node];
}

export function predict(model: ModelData, input: SchoolInput): number {
  const numericFeatures = model.numeric_features;
  const categoricalFeatures = model.categorical_features;

  // 1. Build numeric vector (with imputation)
  const numericValues: number[] = numericFeatures.map((feat, i) => {
    const val = (input as unknown as Record<string, unknown>)[feat] as number | null;
    return val === null || val === undefined || isNaN(val)
      ? model.imputer_medians[i]
      : val;
  });

  // 2. Scale numeric features
  const scaledNumeric = numericValues.map(
    (v, i) => (v - model.scaler_means[i]) / model.scaler_stds[i]
  );

  // 3. One-hot encode categoricals
  const onehotValues: number[] = [];
  categoricalFeatures.forEach((feat, catIdx) => {
    const val = String((input as unknown as Record<string, unknown>)[feat] ?? "missing");
    const categories = model.onehot_categories[catIdx];
    categories.forEach((cat) => {
      onehotValues.push(cat === val ? 1 : 0);
    });
  });

  // 4. Concatenate
  const allFeatures = [...scaledNumeric, ...onehotValues];

  // 5. Average predictions across all trees
  let sum = 0;
  for (const tree of model.trees) {
    sum += traverseTree(tree, allFeatures);
  }
  return sum / model.trees.length;
}
