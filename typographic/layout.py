"""
Layout Analysis Module

Uses OCR bounding boxes to estimate
empty regions where malicious prompts
can be inserted.
"""


class LayoutAnalyzer:

    def __init__(self):
        pass


    def find_empty_regions(self, regions):

        """
        Input:
            OCR regions

        Output:
            Candidate empty regions
        """

        candidates = []

        # Sort by vertical position
        regions = sorted(
            regions,
            key=lambda r: r["polygon"][0][1]
        )

        for i in range(len(regions)-1):

            current = regions[i]

            nxt = regions[i+1]

            current_bottom = max(
                p[1] for p in current["polygon"]
            )

            next_top = min(
                p[1] for p in nxt["polygon"]
            )

            gap = next_top - current_bottom

            if gap > 25:

                candidates.append({

                    "x": 30,

                    "y": current_bottom + 5,

                    "height": gap,

                    "width": 400

                })

        return candidates