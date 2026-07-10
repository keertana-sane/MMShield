from pathlib import Path
import random
import csv

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from ocr import OCRExtractor
from prompts import PROMPTS
from config import TRAIN_IMAGES
from config import ATTACK_OUTPUT
from config import METADATA_OUTPUT


class TypographicAttackGenerator:

    def __init__(self):

        self.ocr = OCRExtractor()

        self.output_folder = ATTACK_OUTPUT

        self.metadata_file = (
            METADATA_OUTPUT /
            "attack_metadata.csv"
        )

        self.colors = [

            (0,0,0),

            (40,40,40),

            (80,80,80),

            (120,120,120),

            (170,170,170)

        ]

    ####################################################

    def choose_prompt(self):

        return random.choice(PROMPTS)

    ####################################################

    def choose_font_size(self):

        return random.randint(8,18)

    ####################################################

    def choose_color(self):

        return random.choice(self.colors)

    ####################################################

    def choose_rotation(self):

        return random.choice(

            [0,10,15,20,30]

        )

    ####################################################

    def load_font(self,size):

        try:

            return ImageFont.truetype(

                "Arial.ttf",

                size

            )

        except:

            return ImageFont.load_default()

    ####################################################

    def find_empty_regions(self,image_path):

        regions = self.ocr.extract_text_regions(

            image_path

        )

        regions = sorted(

            regions,

            key=lambda r:

            min(

                p[1]

                for p in r["polygon"]

            )

        )

        candidates=[]

        for i in range(

            len(regions)-1

        ):

            current=regions[i]

            nxt=regions[i+1]

            current_bottom=max(

                p[1]

                for p in current["polygon"]

            )

            next_top=min(

                p[1]

                for p in nxt["polygon"]

            )

            gap=next_top-current_bottom

            if gap>30:

                candidates.append(

                    {

                        "x":30,

                        "y":current_bottom+5,

                        "width":400,

                        "height":gap

                    }

                )

        return candidates
    
        ####################################################

    def choose_region(self, candidates, image):

        width, height = image.size

        if len(candidates) == 0:

            return {

                "x": 30,

                "y": height - 80,

                "width": width - 60,

                "height": 40

            }

        return random.choice(candidates)

    ####################################################

    def draw_prompt(

        self,

        image,

        prompt,

        region,

        font_size,

        color,

        rotation

    ):

        font = self.load_font(font_size)

        overlay = Image.new(

            "RGBA",

            image.size,

            (255,255,255,0)

        )

        draw = ImageDraw.Draw(overlay)

        draw.text(

            (region["x"], region["y"]),

            prompt,

            font=font,

            fill=color + (255,)

        )

        if rotation != 0:

            overlay = overlay.rotate(

                rotation,

                expand=False

            )

        image = Image.alpha_composite(

            image.convert("RGBA"),

            overlay

        )

        return image.convert("RGB")

    ####################################################

    def save_metadata(

        self,

        image_name,

        prompt,

        font_size,

        color,

        rotation,

        region

    ):

        exists = self.metadata_file.exists()

        with open(

            self.metadata_file,

            "a",

            newline=""

        ) as f:

            writer = csv.writer(f)

            if not exists:

                writer.writerow(

                    [

                        "image",

                        "prompt",

                        "font_size",

                        "color",

                        "rotation",

                        "x",

                        "y"

                    ]

                )

            writer.writerow(

                [

                    image_name,

                    prompt,

                    font_size,

                    color,

                    rotation,

                    region["x"],

                    region["y"]

                ]

            )

    
        ####################################################

    def generate_attack(self,image,image_name,candidates,attack_number,metadata):

     attacked = image.copy()

     region = self.choose_region(

        candidates,

        attacked

     )

     prompt = self.choose_prompt()

     font_size = self.choose_font_size()

     color = self.choose_color()

     rotation = self.choose_rotation()

     attacked = self.draw_prompt(

        attacked,

        prompt,

        region,

        font_size,

        color,

        rotation

     )

     output_name = (

        f"{image_name}_attack_{attack_number}.jpg"

     )

     output_path = (

        self.output_folder /

        output_name

     )

     attacked.save(output_path)

     metadata.append(

        {

            "image": output_name,

            "prompt": prompt,

            "font_size": font_size,

            "color": color,

            "rotation": rotation,

            "x": region["x"],

            "y": region["y"]

        }

     )

     print(

        f"Saved : {output_name}"

     )

    ####################################################

    def generate_dataset(self,num_attacks=5,max_images=20):

     images = sorted(

        TRAIN_IMAGES.glob("*.jpg")

     )

     if max_images is not None:

        images = images[:max_images]

     print(

        f"\nGenerating attacks for {len(images)} receipts...\n"

     )

     metadata = []

     for image_path in images:

        print(

            f"\nProcessing {image_path.name}"

        )

        image = Image.open(

            image_path

        ).convert("RGB")

        image_name = image_path.stem

        candidates = self.find_empty_regions(

            image_path

        )

        for attack in range(

            1,

            num_attacks + 1

        ):

            self.generate_attack(

                image,

                image_name,

                candidates,

                attack,

                metadata

            )

     with open(

        self.metadata_file,

        "w",

        newline=""

     ) as f:

        writer = csv.DictWriter(

            f,

            fieldnames=[

                "image",

                "prompt",

                "font_size",

                "color",

                "rotation",

                "x",

                "y"

            ]

        )

        writer.writeheader()

        writer.writerows(metadata)

     print(

        "\nAttack generation completed."

     )

     print(

        f"Generated {len(metadata)} attack images."

     )


if __name__ == "__main__":

    generator = TypographicAttackGenerator()

    generator.generate_dataset(

    num_attacks=1,

    max_images=None

)