import re

class DocumentStructure:
    def __init__(self, text: str):
        self.original_text = text
        self._lines = text.split('\n')
        self.structure = self._analyze_structure()

    @staticmethod
    def is_heading(line: str) -> bool:
        """Check if a line is a heading - static method for external use"""
        stripped = line.strip()
        if not stripped or len(stripped) > 100:
            return False

        heading_patterns = [
            re.compile(r'^[A-Z\s]{3,}$'),
            re.compile(r'^\d+\.\s+[A-Z]'),
            re.compile(r'^Chapter\s+\d+', re.IGNORECASE),
            re.compile(r'^Section\s+\d+', re.IGNORECASE),
            re.compile(r'^[IVX]+\.\s+'),
        ]

        if any(pattern.match(stripped) for pattern in heading_patterns):
            return True

        return (len(stripped) < 50 and
                not stripped.endswith(('.', '!', '?', ';', ':')) and
                len(stripped.split()) <= 8 and
                any(word[0].isupper() for word in stripped.split() if word))
    
    def _is_heading(self, line: str) -> bool:
        """Instance method that calls the static method"""
        return self.is_heading(line)
    
    def _analyze_structure(self):
        structure = {
            'paragraphs': [],
            'headings': [],
            'total_lines': len(self._lines)
        }

        current_paragraph = []
        paragraph_start = 0

        for i, line in enumerate(self._lines):
            stripped = line.strip()

            if not stripped:  # Empty line
                if current_paragraph:
                    structure['paragraphs'].append({
                        'content': '\n'.join(current_paragraph),
                        'start_line': paragraph_start,
                        'end_line': i - 1,
                        'length': sum(len(l) for l in current_paragraph)
                    })
                    current_paragraph = []
                paragraph_start = i + 1

            elif self._is_heading(line):
                if current_paragraph:
                    structure['paragraphs'].append({
                        'content': '\n'.join(current_paragraph),
                        'start_line': paragraph_start,
                        'end_line': i - 1,
                        'length': sum(len(l) for l in current_paragraph)
                    })
                    current_paragraph = []
                structure['headings'].append({
                    'content': line,
                    'start_line': i,
                    'end_line': i,
                    'length': len(line)
                })
                paragraph_start = i + 1

            else:
                current_paragraph.append(line)

        if current_paragraph:
            structure['paragraphs'].append({
                'content': '\n'.join(current_paragraph),
                'start_line': paragraph_start,
                'end_line': len(self._lines) - 1,
                'length': sum(len(l) for l in current_paragraph)
            })

        return structure

    def _get_heading_level(self, line: str) -> int:
        stripped = line.strip()
        if re.match(r'^[A-Z\s]{3,}$', stripped):
            return 1
        elif re.match(r'^\d+\.\s+', stripped):
            return 2
        else:
            return 3