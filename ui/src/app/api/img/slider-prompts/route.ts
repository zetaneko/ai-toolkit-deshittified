import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';
import { writeSliderPromptOverrides } from '@/server/sliderPrompts';

function isUnderRoot(filepath: string, root: string): boolean {
  const resolved = path.resolve(filepath);
  const resolvedRoot = path.resolve(root);
  return resolved === resolvedRoot || resolved.startsWith(resolvedRoot + path.sep);
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { imgPath, positive_prompt, negative_prompt } = body ?? {};
    if (typeof imgPath !== 'string' || imgPath === '') {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }

    const datasetsPath = await getDatasetsRoot();
    if (!isUnderRoot(imgPath, datasetsPath)) {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }

    try {
      await fs.promises.access(imgPath);
    } catch {
      return NextResponse.json({ error: 'Image does not exist' }, { status: 404 });
    }

    await writeSliderPromptOverrides(
      imgPath,
      typeof positive_prompt === 'string' ? positive_prompt : '',
      typeof negative_prompt === 'string' ? negative_prompt : '',
    );
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error saving slider prompts:', error);
    return NextResponse.json({ error: 'Failed to save slider prompts' }, { status: 500 });
  }
}
