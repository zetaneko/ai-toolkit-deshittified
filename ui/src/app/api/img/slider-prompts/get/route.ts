import { NextResponse } from 'next/server';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';
import { readSliderPromptOverrides } from '@/server/sliderPrompts';

function isUnderRoot(filepath: string, root: string): boolean {
  const resolved = path.resolve(filepath);
  const resolvedRoot = path.resolve(root);
  return resolved === resolvedRoot || resolved.startsWith(resolvedRoot + path.sep);
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { imgPath } = body ?? {};
    if (typeof imgPath !== 'string' || imgPath === '') {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }

    const datasetsPath = await getDatasetsRoot();
    if (!isUnderRoot(imgPath, datasetsPath)) {
      return NextResponse.json({ error: 'Access denied' }, { status: 403 });
    }

    const prompts = await readSliderPromptOverrides(imgPath);
    return NextResponse.json(prompts);
  } catch (error) {
    console.error('Error getting slider prompts:', error);
    const message = error instanceof Error ? error.message : 'Failed to get slider prompts';
    const status = message.startsWith('Invalid concept slider') || message.startsWith('Concept slider prompt file') ? 400 : 500;
    return NextResponse.json({ error: message }, { status });
  }
}
