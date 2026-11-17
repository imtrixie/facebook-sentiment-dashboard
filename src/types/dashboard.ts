export type Sentiment = 'positive' | 'neutral' | 'negative';

export type ModelType = 'Naive Bayes' | 'Decision Tree';

export type VectorizerType = 'TF-IDF' | 'Bag of Words';

export interface FacebookPost {
  post_id: string;
  timestamp: string;
  timestampMs: number;
  text: string;
  category: string;
  sentiment: Sentiment;
  language: string;
  has_link: number;
  likes: number;
  comments: number;
  shares: number;
  is_spam: number;
}

export interface PredictedPost extends FacebookPost {
  predicted_sentiment: Sentiment;
  is_correct: boolean;
}

export interface ClassMetrics {
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface ModelEvaluation {
  modelName: ModelType;
  testAccuracy: number;
  testMacroF1: number;
  trainSize: number;
  testSize: number;
  classMetrics: Record<Sentiment, ClassMetrics>;
  confusionMatrix: number[][];
  oofPredictions: Sentiment[];
}

export interface PredictionOutput {
  label: Sentiment;
  probabilities: Record<Sentiment, number>;
}

export interface FilterState {
  categories: string[];
  sentiments: Sentiment[];
  languages: string[];
  excludeSpam: boolean;
}
