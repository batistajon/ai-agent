from app.frameworks.routes import router
from app.domain.interfaces.IFramework import IFramework
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


class FastAPIFrameworkSingleton(IFramework):
    """
    Concretion for FastAPI framework

    Params: IFramework
    Returns: A singleton instance of the framework
    """
    def get_instance(self) -> FastAPI:
        fastapi_singleton = FastAPI(title="Modular AI Agent")

        origins = [
            "http://localhost.com",
            "https://localhost.com",
            "http://localhost",
            "http://localhost:5000",
            "http://localhost:8501",
            "http://localhost:8501",
            "http://localhost:8000",
        ]

        fastapi_singleton.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        return fastapi_singleton


    def load_routes(self, instance: object):
        return instance.include_router(router.router)
