import {
  ClassMetrics,
  FacebookPost,
  ModelEvaluation,
  ModelType,
  PredictionOutput,
  Sentiment,
  VectorizerType,
} from '../types/dashboard';

export const SENTIMENT_CLASSES: Sentiment[] = ['positive', 'neutral', 'negative'];

const CLASS_TO_IDX: Record<Sentiment, number> = {
  positive: 0,
  neutral: 1,
  negative: 2,
};

function createSeededRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffleIndices(indices: number[], seed: number): number[] {
  const arr = [...indices];
  const rand = createSeededRng(seed);
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

export function tokenizeText(text: string): string[] {
  const normalized = text.toLowerCase().trim();
  if (!normalized) return [];

  // Tách từ, hashtag và biểu tượng cảm xúc (emoji)
  const rawTokens = normalized.match(/#[\p{L}\p{N}_]+|[\p{L}\p{N}]+|[\p{Emoji_Presentation}\p{Extended_Pictographic}]/gu) ?? [];
  if (rawTokens.length === 0) return [];

  const tokens = [...rawTokens];
  for (let i = 0; i < rawTokens.length - 1; i++) {
    tokens.push(`${rawTokens[i]}_${rawTokens[i + 1]}`);
  }
  return tokens;
}

interface SparseVector {
  indices: number[];
  values: number[];
}

interface FittedVectorizer {
  vocabSize: number;
  vocabTerms: string[];
  idf: Float64Array;
  vectorizerType: VectorizerType;
  transform: (texts: string[]) => SparseVector[];
}

function fitVectorizer(
  texts: string[],
  vectorizerType: VectorizerType,
  maxFeatures = 1800
): FittedVectorizer {
  const docFreq = new Map<string, number>();
  const tokenizedDocs = texts.map((t) => tokenizeText(t));

  for (const tokens of tokenizedDocs) {
    const seen = new Set(tokens);
    for (const tok of seen) {
      docFreq.set(tok, (docFreq.get(tok) ?? 0) + 1);
    }
  }

  // Lọc các token xuất hiện tối thiểu 2 lần và chọn top maxFeatures phổ biến nhất
  const sortedTerms = Array.from(docFreq.entries())
    .filter(([, df]) => df >= 2)
    .sort((a, b) => b[1] - a[1])
    .slice(0, maxFeatures)
    .map(([term]) => term);

  const termToIdx = new Map<string, number>();
  sortedTerms.forEach((term, idx) => termToIdx.set(term, idx));

  const nDocs = Math.max(texts.length, 1);
  const idf = new Float64Array(sortedTerms.length);
  for (let i = 0; i < sortedTerms.length; i++) {
    const df = docFreq.get(sortedTerms[i]) ?? 1;
    idf[i] = Math.log((nDocs + 1) / (df + 1)) + 1;
  }

  const transform = (inputTexts: string[]): SparseVector[] => {
    return inputTexts.map((text) => {
      const tokens = tokenizeText(text);
      const counts = new Map<number, number>();
      for (const tok of tokens) {
        const idx = termToIdx.get(tok);
        if (idx !== undefined) {
          counts.set(idx, (counts.get(idx) ?? 0) + 1);
        }
      }

      const indices = Array.from(counts.keys()).sort((a, b) => a - b);
      const values = new Array<number>(indices.length);

      if (vectorizerType === 'Bag of Words') {
        for (let k = 0; k < indices.length; k++) {
          values[k] = counts.get(indices[k]) ?? 0;
        }
      } else {
        let normSq = 0;
        for (let k = 0; k < indices.length; k++) {
          const idx = indices[k];
          const tf = counts.get(idx) ?? 0;
          const sublinearTf = 1 + Math.log(tf);
          const w = sublinearTf * idf[idx];
          values[k] = w;
          normSq += w * w;
        }
        const norm = Math.sqrt(normSq);
        if (norm > 0) {
          for (let k = 0; k < values.length; k++) {
            values[k] /= norm;
          }
        }
      }

      return { indices, values };
    });
  };

  return {
    vocabSize: sortedTerms.length,
    vocabTerms: sortedTerms,
    idf,
    vectorizerType,
    transform,
  };
}

// ============================================================================
// MÔ HÌNH 1: MULTINOMIAL NAIVE BAYES
// ============================================================================
class MultinomialNaiveBayesModel {
  private logPriors = new Float64Array(3);
  private logLikelihoods: Float64Array[] = [];
  private vocabSize = 0;
  private alpha = 0.5;

  constructor(alpha = 0.5) {
    this.alpha = alpha;
  }

  fit(vectors: SparseVector[], labels: Sentiment[], vocabSize: number): void {
    this.vocabSize = Math.max(vocabSize, 1);
    const classCounts = new Float64Array(3);
    const featureSums = [
      new Float64Array(this.vocabSize),
      new Float64Array(this.vocabSize),
      new Float64Array(this.vocabSize),
    ];
    const totalWeightsPerClass = new Float64Array(3);

    for (let i = 0; i < vectors.length; i++) {
      const c = CLASS_TO_IDX[labels[i]];
      classCounts[c] += 1;
      const vec = vectors[i];
      for (let k = 0; k < vec.indices.length; k++) {
        const featIdx = vec.indices[k];
        const val = vec.values[k];
        featureSums[c][featIdx] += val;
        totalWeightsPerClass[c] += val;
      }
    }

    const totalDocs = Math.max(vectors.length, 1);
    this.logLikelihoods = [
      new Float64Array(this.vocabSize),
      new Float64Array(this.vocabSize),
      new Float64Array(this.vocabSize),
    ];

    for (let c = 0; c < 3; c++) {
      this.logPriors[c] = Math.log((classCounts[c] + 1e-6) / totalDocs);
      const denom = totalWeightsPerClass[c] + this.alpha * this.vocabSize;
      for (let f = 0; f < this.vocabSize; f++) {
        this.logLikelihoods[c][f] = Math.log((featureSums[c][f] + this.alpha) / denom);
      }
    }
  }

  predictProba(vectors: SparseVector[]): PredictionOutput[] {
    return vectors.map((vec) => {
      const logProbs = [this.logPriors[0], this.logPriors[1], this.logPriors[2]];
      for (let k = 0; k < vec.indices.length; k++) {
        const featIdx = vec.indices[k];
        const val = vec.values[k];
        if (featIdx < this.vocabSize) {
          logProbs[0] += val * this.logLikelihoods[0][featIdx];
          logProbs[1] += val * this.logLikelihoods[1][featIdx];
          logProbs[2] += val * this.logLikelihoods[2][featIdx];
        }
      }

      const maxLog = Math.max(logProbs[0], logProbs[1], logProbs[2]);
      const exps = [
        Math.exp(logProbs[0] - maxLog),
        Math.exp(logProbs[1] - maxLog),
        Math.exp(logProbs[2] - maxLog),
      ];
      const sumExp = exps[0] + exps[1] + exps[2];
      const probs = [exps[0] / sumExp, exps[1] / sumExp, exps[2] / sumExp];

      let bestIdx = 0;
      if (probs[1] > probs[bestIdx]) bestIdx = 1;
      if (probs[2] > probs[bestIdx]) bestIdx = 2;

      return {
        label: SENTIMENT_CLASSES[bestIdx],
        probabilities: {
          positive: probs[0],
          neutral: probs[1],
          negative: probs[2],
        },
      };
    });
  }
}

// ============================================================================
// MÔ HÌNH 2: DECISION TREE CLASSIFIER (CART GINI + BALANCED CLASS WEIGHTS)
// ============================================================================
interface TreeNode {
  featureIdx?: number;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
  probabilities: [number, number, number];
  predictedIdx: number;
}

class DecisionTreeModel {
  private root: TreeNode | null = null;
  private maxDepth: number;
  private minSamplesSplit: number;
  private maxCandidateFeatures: number;

  constructor(maxDepth = 16, minSamplesSplit = 8, maxCandidateFeatures = 260) {
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
    this.maxCandidateFeatures = maxCandidateFeatures;
  }

  fit(vectors: SparseVector[], labels: Sentiment[], vocabSize: number): void {
    const nDocs = vectors.length;
    if (nDocs === 0) return;

    const labelIndices = new Int8Array(nDocs);
    const classCounts = [0, 0, 0];
    for (let i = 0; i < nDocs; i++) {
      const idx = CLASS_TO_IDX[labels[i]];
      labelIndices[i] = idx;
      classCounts[idx]++;
    }

    // Trọng số cân bằng lớp (class_weight = balanced)
    const classWeights: [number, number, number] = [
      nDocs / (3 * Math.max(classCounts[0], 1)),
      nDocs / (3 * Math.max(classCounts[1], 1)),
      nDocs / (3 * Math.max(classCounts[2], 1)),
    ];

    // Chọn tập đặc trưng ứng viên có độ phân tách lớp tốt nhất (variance / chi2 proxy)
    const candidateFeatures = this.selectCandidateFeatures(vectors, labelIndices, vocabSize);

    // Xây dựng bản đồ truy cập nhanh theo đặc trưng: feature -> doc indices & values
    const docFeatureMaps = vectors.map((vec) => {
      const map = new Map<number, number>();
      for (let k = 0; k < vec.indices.length; k++) {
        map.set(vec.indices[k], vec.values[k]);
      }
      return map;
    });

    const allDocIndices = Array.from({ length: nDocs }, (_, i) => i);
    this.root = this.buildNode(
      allDocIndices,
      docFeatureMaps,
      labelIndices,
      classWeights,
      candidateFeatures,
      0
    );
  }

  private selectCandidateFeatures(
    vectors: SparseVector[],
    labelIndices: Int8Array,
    vocabSize: number
  ): number[] {
    const featClassCounts = [
      new Float64Array(vocabSize),
      new Float64Array(vocabSize),
      new Float64Array(vocabSize),
    ];
    const featTotal = new Float64Array(vocabSize);

    for (let i = 0; i < vectors.length; i++) {
      const c = labelIndices[i];
      const vec = vectors[i];
      for (let k = 0; k < vec.indices.length; k++) {
        const f = vec.indices[k];
        featClassCounts[c][f] += 1;
        featTotal[f] += 1;
      }
    }

    const scored: { f: number; score: number }[] = [];
    for (let f = 0; f < vocabSize; f++) {
      const tot = featTotal[f];
      if (tot < 3) continue;
      const p0 = featClassCounts[0][f] / tot;
      const p1 = featClassCounts[1][f] / tot;
      const p2 = featClassCounts[2][f] / tot;
      const purity = Math.max(p0, p1, p2);
      const score = purity * Math.log(1 + tot);
      scored.push({ f, score });
    }

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, this.maxCandidateFeatures).map((s) => s.f);
  }

  private computeWeightedCounts(
    docIndices: number[],
    labelIndices: Int8Array,
    classWeights: [number, number, number]
  ): { weights: [number, number, number]; total: number; probs: [number, number, number]; predIdx: number } {
    const weights: [number, number, number] = [0, 0, 0];
    for (const docIdx of docIndices) {
      const c = labelIndices[docIdx];
      weights[c] += classWeights[c];
    }
    const total = weights[0] + weights[1] + weights[2];
    const probs: [number, number, number] =
      total > 0
        ? [weights[0] / total, weights[1] / total, weights[2] / total]
        : [1 / 3, 1 / 3, 1 / 3];

    let predIdx = 0;
    if (probs[1] > probs[predIdx]) predIdx = 1;
    if (probs[2] > probs[predIdx]) predIdx = 2;

    return { weights, total, probs, predIdx };
  }

  private giniFromWeights(w: [number, number, number], total: number): number {
    if (total <= 0) return 0;
    const p0 = w[0] / total;
    const p1 = w[1] / total;
    const p2 = w[2] / total;
    return 1 - (p0 * p0 + p1 * p1 + p2 * p2);
  }

  private buildNode(
    docIndices: number[],
    docFeatureMaps: Map<number, number>[],
    labelIndices: Int8Array,
    classWeights: [number, number, number],
    candidateFeatures: number[],
    depth: number
  ): TreeNode {
    const stats = this.computeWeightedCounts(docIndices, labelIndices, classWeights);
    const parentGini = this.giniFromWeights(stats.weights, stats.total);

    if (
      depth >= this.maxDepth ||
      docIndices.length < this.minSamplesSplit ||
      parentGini <= 1e-6
    ) {
      return {
        probabilities: stats.probs,
        predictedIdx: stats.predIdx,
      };
    }

    let bestGain = 0;
    let bestFeature = -1;
    let bestThreshold = 0;

    for (const f of candidateFeatures) {
      // Nhanh: tách nhị phân feature <= 0 (không xuất hiện) vs > 0 (xuất hiện)
      // Đây là phép tách tối ưu và chuẩn cho dữ liệu văn bản thưa (TF-IDF / BoW)
      const rightWeights: [number, number, number] = [0, 0, 0];
      let rightCount = 0;

      for (const docIdx of docIndices) {
        if (docFeatureMaps[docIdx].has(f)) {
          const c = labelIndices[docIdx];
          rightWeights[c] += classWeights[c];
          rightCount++;
        }
      }

      if (rightCount < 2 || docIndices.length - rightCount < 2) continue;

      const rightTotal = rightWeights[0] + rightWeights[1] + rightWeights[2];
      const leftWeights: [number, number, number] = [
        stats.weights[0] - rightWeights[0],
        stats.weights[1] - rightWeights[1],
        stats.weights[2] - rightWeights[2],
      ];
      const leftTotal = stats.total - rightTotal;

      if (leftTotal <= 0 || rightTotal <= 0) continue;

      const weightedGini =
        (leftTotal / stats.total) * this.giniFromWeights(leftWeights, leftTotal) +
        (rightTotal / stats.total) * this.giniFromWeights(rightWeights, rightTotal);
      const gain = parentGini - weightedGini;

      if (gain > bestGain) {
        bestGain = gain;
        bestFeature = f;
        bestThreshold = 0;
      }
    }

    if (bestFeature === -1 || bestGain <= 1e-5) {
      return {
        probabilities: stats.probs,
        predictedIdx: stats.predIdx,
      };
    }

    const leftDocs: number[] = [];
    const rightDocs: number[] = [];
    for (const docIdx of docIndices) {
      if ((docFeatureMaps[docIdx].get(bestFeature) ?? 0) > bestThreshold) {
        rightDocs.push(docIdx);
      } else {
        leftDocs.push(docIdx);
      }
    }

    if (leftDocs.length === 0 || rightDocs.length === 0) {
      return {
        probabilities: stats.probs,
        predictedIdx: stats.predIdx,
      };
    }

    return {
      featureIdx: bestFeature,
      threshold: bestThreshold,
      left: this.buildNode(
        leftDocs,
        docFeatureMaps,
        labelIndices,
        classWeights,
        candidateFeatures,
        depth + 1
      ),
      right: this.buildNode(
        rightDocs,
        docFeatureMaps,
        labelIndices,
        classWeights,
        candidateFeatures,
        depth + 1
      ),
      probabilities: stats.probs,
      predictedIdx: stats.predIdx,
    };
  }

  predictProba(vectors: SparseVector[]): PredictionOutput[] {
    if (!this.root) {
      return vectors.map(() => ({
        label: 'neutral',
        probabilities: { positive: 1 / 3, neutral: 1 / 3, negative: 1 / 3 },
      }));
    }

    return vectors.map((vec) => {
      const featMap = new Map<number, number>();
      for (let k = 0; k < vec.indices.length; k++) {
        featMap.set(vec.indices[k], vec.values[k]);
      }

      let node = this.root!;
      while (node.featureIdx !== undefined && node.left && node.right) {
        const val = featMap.get(node.featureIdx) ?? 0;
        node = val > (node.threshold ?? 0) ? node.right : node.left;
      }

      return {
        label: SENTIMENT_CLASSES[node.predictedIdx],
        probabilities: {
          positive: node.probabilities[0],
          neutral: node.probabilities[1],
          negative: node.probabilities[2],
        },
      };
    });
  }
}

// ============================================================================
// CHIA TRAIN/TEST STRATIFIED & TÍNH TOÁN CHỈ SỐ ĐÁNH GIÁ
// ============================================================================
function stratifiedTrainTestSplit(
  labels: Sentiment[],
  testRatio = 0.2,
  seed = 42
): { trainIdx: number[]; testIdx: number[] } {
  const byClass: Record<Sentiment, number[]> = {
    positive: [],
    neutral: [],
    negative: [],
  };

  labels.forEach((lbl, idx) => byClass[lbl].push(idx));

  const trainIdx: number[] = [];
  const testIdx: number[] = [];

  for (const cls of SENTIMENT_CLASSES) {
    const shuffled = shuffleIndices(byClass[cls], seed + CLASS_TO_IDX[cls] * 17);
    const nTest = Math.max(1, Math.round(shuffled.length * testRatio));
    testIdx.push(...shuffled.slice(0, nTest));
    trainIdx.push(...shuffled.slice(nTest));
  }

  return { trainIdx, testIdx };
}

function createStratifiedFolds(labels: Sentiment[], k = 5, seed = 42): number[][] {
  const byClass: Record<Sentiment, number[]> = {
    positive: [],
    neutral: [],
    negative: [],
  };
  labels.forEach((lbl, idx) => byClass[lbl].push(idx));

  const folds: number[][] = Array.from({ length: k }, () => []);
  for (const cls of SENTIMENT_CLASSES) {
    const shuffled = shuffleIndices(byClass[cls], seed + CLASS_TO_IDX[cls] * 31);
    shuffled.forEach((idx, pos) => {
      folds[pos % k].push(idx);
    });
  }
  return folds;
}

function evaluatePredictions(
  yTrue: Sentiment[],
  yPred: Sentiment[]
): {
  accuracy: number;
  macroF1: number;
  classMetrics: Record<Sentiment, ClassMetrics>;
  confusionMatrix: number[][];
} {
  const cm = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];

  let correct = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const t = CLASS_TO_IDX[yTrue[i]];
    const p = CLASS_TO_IDX[yPred[i]];
    cm[t][p] += 1;
    if (t === p) correct++;
  }

  const accuracy = yTrue.length > 0 ? correct / yTrue.length : 0;
  const classMetrics = {} as Record<Sentiment, ClassMetrics>;
  let f1Sum = 0;

  for (const cls of SENTIMENT_CLASSES) {
    const idx = CLASS_TO_IDX[cls];
    const tp = cm[idx][idx];
    let fp = 0;
    let fn = 0;
    let support = 0;

    for (let r = 0; r < 3; r++) {
      support += cm[idx][r];
      if (r !== idx) {
        fp += cm[r][idx];
        fn += cm[idx][r];
      }
    }

    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

    classMetrics[cls] = { precision, recall, f1, support };
    f1Sum += f1;
  }

  return {
    accuracy,
    macroF1: f1Sum / 3,
    classMetrics,
    confusionMatrix: cm,
  };
}

export interface TrainedMlSuite {
  vectorizerType: VectorizerType;
  evaluations: Record<ModelType, ModelEvaluation>;
  predictText: (modelName: ModelType, text: string) => PredictionOutput;
}

export function trainMlSuite(
  posts: FacebookPost[],
  vectorizerType: VectorizerType
): TrainedMlSuite {
  const texts = posts.map((p) => p.text);
  const labels = posts.map((p) => p.sentiment);

  // 1. Chia tập Train/Test (80/20) cho đánh giá độc lập & dự đoán trực tiếp
  const { trainIdx, testIdx } = stratifiedTrainTestSplit(labels, 0.2, 42);
  const trainTexts = trainIdx.map((i) => texts[i]);
  const trainLabels = trainIdx.map((i) => labels[i]);
  const testTexts = testIdx.map((i) => texts[i]);
  const testLabels = testIdx.map((i) => labels[i]);

  const holdoutVectorizer = fitVectorizer(trainTexts, vectorizerType);
  const xTrainVecs = holdoutVectorizer.transform(trainTexts);
  const xTestVecs = holdoutVectorizer.transform(testTexts);

  const nbHoldout = new MultinomialNaiveBayesModel(0.5);
  nbHoldout.fit(xTrainVecs, trainLabels, holdoutVectorizer.vocabSize);
  const nbTestPreds = nbHoldout.predictProba(xTestVecs).map((o) => o.label);
  const nbMetrics = evaluatePredictions(testLabels, nbTestPreds);

  const dtHoldout = new DecisionTreeModel(16, 8, 220);
  dtHoldout.fit(xTrainVecs, trainLabels, holdoutVectorizer.vocabSize);
  const dtTestPreds = dtHoldout.predictProba(xTestVecs).map((o) => o.label);
  const dtMetrics = evaluatePredictions(testLabels, dtTestPreds);

  // 2. Dự đoán ngoài mẫu (5-Fold Out-of-Fold) cho toàn bộ bảng dữ liệu
  // Đảm bảo không dùng nhãn thật làm dự đoán và không dự đoán trên dữ liệu đã học
  const folds = createStratifiedFolds(labels, 5, 42);
  const nbOof = new Array<Sentiment>(posts.length);
  const dtOof = new Array<Sentiment>(posts.length);

  for (let f = 0; f < folds.length; f++) {
    const valSet = new Set(folds[f]);
    const foldTrainTexts: string[] = [];
    const foldTrainLabels: Sentiment[] = [];
    const foldValIdx: number[] = [];
    const foldValTexts: string[] = [];

    for (let i = 0; i < posts.length; i++) {
      if (valSet.has(i)) {
        foldValIdx.push(i);
        foldValTexts.push(texts[i]);
      } else {
        foldTrainTexts.push(texts[i]);
        foldTrainLabels.push(labels[i]);
      }
    }

    const foldVec = fitVectorizer(foldTrainTexts, vectorizerType, 1400);
    const foldXTrain = foldVec.transform(foldTrainTexts);
    const foldXVal = foldVec.transform(foldValTexts);

    const nbFold = new MultinomialNaiveBayesModel(0.5);
    nbFold.fit(foldXTrain, foldTrainLabels, foldVec.vocabSize);
    const nbValOut = nbFold.predictProba(foldXVal);

    const dtFold = new DecisionTreeModel(14, 10, 160);
    dtFold.fit(foldXTrain, foldTrainLabels, foldVec.vocabSize);
    const dtValOut = dtFold.predictProba(foldXVal);

    foldValIdx.forEach((origIdx, pos) => {
      nbOof[origIdx] = nbValOut[pos].label;
      dtOof[origIdx] = dtValOut[pos].label;
    });
  }

  return {
    vectorizerType,
    evaluations: {
      'Naive Bayes': {
        modelName: 'Naive Bayes',
        testAccuracy: nbMetrics.accuracy,
        testMacroF1: nbMetrics.macroF1,
        trainSize: trainIdx.length,
        testSize: testIdx.length,
        classMetrics: nbMetrics.classMetrics,
        confusionMatrix: nbMetrics.confusionMatrix,
        oofPredictions: nbOof,
      },
      'Decision Tree': {
        modelName: 'Decision Tree',
        testAccuracy: dtMetrics.accuracy,
        testMacroF1: dtMetrics.macroF1,
        trainSize: trainIdx.length,
        testSize: testIdx.length,
        classMetrics: dtMetrics.classMetrics,
        confusionMatrix: dtMetrics.confusionMatrix,
        oofPredictions: dtOof,
      },
    },
    predictText: (modelName: ModelType, inputText: string): PredictionOutput => {
      const vecs = holdoutVectorizer.transform([inputText]);
      if (modelName === 'Naive Bayes') {
        return nbHoldout.predictProba(vecs)[0];
      }
      return dtHoldout.predictProba(vecs)[0];
    },
  };
}
