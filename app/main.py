from app.infrastructure.frameworks.patterns.singleton import FrameworkSingleton
from app.infrastructure.frameworks.domain.entities.FastAPI import FastAPIFrameworkSingleton
import uvicorn

framework = FrameworkSingleton(FastAPIFrameworkSingleton())
app = framework.get_instance()
framework.load_routes(app)

if __name__ == "__main__":
    uvicorn.run("main:app", port=5000, log_level="info")

