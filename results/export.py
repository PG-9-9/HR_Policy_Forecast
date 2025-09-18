"""
Export Management Module
Handle different export formats and data serialization.
"""

from enum import Enum
from typing import Any, Dict
from pathlib import Path
import pandas as pd
import json


class ExportFormat(Enum):
    """Supported export formats."""
    CSV = "csv"
    JSON = "json"
    EXCEL = "xlsx"
    HTML = "html"


class ExportManager:
    """Manage data export in various formats."""
    
    def __init__(self, output_dir: str = "results"):
        """Initialize export manager."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_dataframe(self, 
                        df: pd.DataFrame, 
                        filename: str, 
                        format: ExportFormat = ExportFormat.CSV):
        """Export DataFrame to specified format."""
        filepath = self.output_dir / f"{filename}.{format.value}"
        
        if format == ExportFormat.CSV:
            df.to_csv(filepath, index=False)
        elif format == ExportFormat.JSON:
            df.to_json(filepath, orient='records', indent=2)
        elif format == ExportFormat.EXCEL:
            df.to_excel(filepath, index=False)
        elif format == ExportFormat.HTML:
            df.to_html(filepath, index=False)
        
        return str(filepath)
    
    def export_dict(self, 
                   data: Dict[str, Any], 
                   filename: str,
                   format: ExportFormat = ExportFormat.JSON):
        """Export dictionary to specified format."""
        filepath = self.output_dir / f"{filename}.{format.value}"
        
        if format == ExportFormat.JSON:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        return str(filepath)