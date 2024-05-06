from pydantic import BaseModel
from typing import List

class FilePDF(BaseModel):
   title: str
   text: List[str]
