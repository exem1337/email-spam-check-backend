import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.linear_model import LogisticRegression
import uvicorn

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


class SpamFilter:
    def __init__(self):
        self.model = None
        self.classifier = None

    def train_model(self, data: List[str], labels: List[int]):
        logger.info("Начало обучения модели")

        # Подготовка данных для Doc2Vec
        tagged_data = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(data)]
        logger.info("Данные подготовлены для Doc2Vec")

        # Обучение модели Doc2Vec
        self.model = Doc2Vec(vector_size=100, alpha=0.025, min_alpha=0.025, min_count=1, dm=1)
        self.model.build_vocab(tagged_data)
        logger.info("Словарь построен")

        for epoch in range(100):
            self.model.train(tagged_data, total_examples=self.model.corpus_count, epochs=self.model.epochs)
            self.model.alpha -= 0.0002
            self.model.min_alpha = self.model.alpha
            if epoch % 10 == 0:
                logger.info(f"Эпоха {epoch} завершена")

        logger.info("Обучение модели Doc2Vec завершено")

        # Преобразование текстов в векторы
        vectors = [self.model.infer_vector(text.split()) for text in data]
        logger.info("Тексты преобразованы в векторы")

        # Обучение классификатора
        self.classifier = LogisticRegression()
        self.classifier.fit(vectors, labels)
        logger.info("Классификатор обучен")

    def predict(self, text: str) -> int:
        logger.info(f"Получение предсказания для текста: {text}")
        vector = self.model.infer_vector(text.split())
        prediction = self.classifier.predict([vector])
        logger.info(f"Предсказание: {prediction[0]}")
        return int(prediction[0])


spam_filter = SpamFilter()


class TrainRequest(BaseModel):
    data: List[str]
    labels: List[int]


class PredictRequest(BaseModel):
    text: str


@app.post("/train")
async def train_model(request: TrainRequest):
    try:
        spam_filter.train_model(request.data, request.labels)
        return {"message": "Model trained successfully"}
    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        prediction = spam_filter.predict(request.text)
        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Запуск сервера без multiprocessing
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", workers=1)