import fs from 'fs';

/**
 * Per-image concept slider prompts.
 * Keep the file rules in sync with extensions_built_in/concept_slider/slider_prompts.py.
 *
 * photo.slider.json owns both sides while it exists. Otherwise photo.pos.txt and
 * photo.neg.txt each override one side. A blank side means "use the job prompt".
 */

export function imageStem(imgPath: string): string {
  return imgPath.replace(/\.[^/.]+$/, '');
}

export function sliderSidecarPaths(imgPath: string): { jsonPath: string; posPath: string; negPath: string } {
  const stem = imageStem(imgPath);
  return {
    jsonPath: stem + '.slider.json',
    posPath: stem + '.pos.txt',
    negPath: stem + '.neg.txt',
  };
}

function blankToNull(value: unknown): string | null {
  if (value == null) return null;
  const text = String(value).trim();
  return text === '' ? null : text;
}

async function readTextOverride(filePath: string): Promise<string | null> {
  try {
    const text = await fs.promises.readFile(filePath, 'utf-8');
    return blankToNull(text);
  } catch (err: any) {
    if (err?.code === 'ENOENT') return null;
    throw err;
  }
}

function pickField(data: Record<string, unknown>, primary: string, alias: string): string | null {
  if (Object.prototype.hasOwnProperty.call(data, primary)) {
    return blankToNull(data[primary]);
  }
  if (Object.prototype.hasOwnProperty.call(data, alias)) {
    return blankToNull(data[alias]);
  }
  return null;
}

export async function readSliderPromptOverrides(
  imgPath: string,
): Promise<{ positive_prompt: string; negative_prompt: string }> {
  const { jsonPath, posPath, negPath } = sliderSidecarPaths(imgPath);
  try {
    const raw = await fs.promises.readFile(jsonPath, 'utf-8');
    let data: unknown;
    try {
      data = JSON.parse(raw);
    } catch {
      throw new Error(`Invalid concept slider prompt file ${jsonPath}`);
    }
    if (data == null || typeof data !== 'object' || Array.isArray(data)) {
      throw new Error(`Concept slider prompt file ${jsonPath} must be a JSON object`);
    }
    const record = data as Record<string, unknown>;
    return {
      positive_prompt: pickField(record, 'positive_prompt', 'positive') ?? '',
      negative_prompt: pickField(record, 'negative_prompt', 'negative') ?? '',
    };
  } catch (err: any) {
    if (err?.code !== 'ENOENT') throw err;
  }

  const [positive, negative] = await Promise.all([readTextOverride(posPath), readTextOverride(negPath)]);
  return {
    positive_prompt: positive ?? '',
    negative_prompt: negative ?? '',
  };
}

export async function writeSliderPromptOverrides(
  imgPath: string,
  positivePrompt: string,
  negativePrompt: string,
): Promise<void> {
  const { jsonPath, posPath, negPath } = sliderSidecarPaths(imgPath);
  const positive = positivePrompt.trim();
  const negative = negativePrompt.trim();

  // The viewer save is the source of truth, so drop the plain-text sidecars.
  await Promise.all(
    [posPath, negPath].map(async filePath => {
      try {
        await fs.promises.unlink(filePath);
      } catch (err: any) {
        if (err?.code !== 'ENOENT') throw err;
      }
    }),
  );

  if (positive === '' && negative === '') {
    try {
      await fs.promises.unlink(jsonPath);
    } catch (err: any) {
      if (err?.code !== 'ENOENT') throw err;
    }
    return;
  }

  const body = JSON.stringify(
    {
      positive_prompt: positive,
      negative_prompt: negative,
    },
    null,
    2,
  );
  await fs.promises.writeFile(jsonPath, body + '\n', 'utf-8');
}

export async function deleteSliderPromptSidecars(imgPath: string): Promise<void> {
  const { jsonPath, posPath, negPath } = sliderSidecarPaths(imgPath);
  await Promise.all(
    [jsonPath, posPath, negPath].map(async filePath => {
      try {
        await fs.promises.unlink(filePath);
      } catch (err: any) {
        if (err?.code !== 'ENOENT') throw err;
      }
    }),
  );
}
