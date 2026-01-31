/**
 * Static dictionary support for Delta LTSC.
 *
 * Pre-built dictionaries for common domains that can be used
 * to improve compression of domain-specific content.
 */

/**
 * Static dictionary definition.
 */
export interface StaticDictionary {
  /**
   * Unique identifier for the dictionary.
   */
  id: string;

  /**
   * Dictionary version.
   */
  version: string;

  /**
   * Human-readable name.
   */
  name: string;

  /**
   * Description of what this dictionary is optimized for.
   */
  description: string;

  /**
   * Dictionary entries: meta-token ID -> definition tokens.
   */
  entries: Map<number, readonly number[]>;

  /**
   * Patterns (token subsequences) to definitions.
   */
  patterns: Map<string, number>;
}

// Import dictionaries directly (with import attributes for Node.js 22+ compatibility)
import pythonDict from './python.json' with { type: 'json' };
import typescriptDict from './typescript.json' with { type: 'json' };
import markdownDict from './markdown.json' with { type: 'json' };
import jsonDict from './json.json' with { type: 'json' };
import sqlDict from './sql.json' with { type: 'json' };

/**
 * Available built-in static dictionaries.
 */
export const STATIC_DICTIONARIES = {
  'python-v1': pythonDict,
  'typescript-v1': typescriptDict,
  'markdown-v1': markdownDict,
  'json-v1': jsonDict,
  'sql-v1': sqlDict,
} as const;

export type StaticDictionaryId = keyof typeof STATIC_DICTIONARIES;

/**
 * Raw dictionary JSON format.
 */
interface DictionaryJson {
  id: string;
  version: string;
  name: string;
  description: string;
  entries: Array<{
    metaToken: number;
    pattern: number[];
  }>;
}

/**
 * Load a built-in static dictionary.
 *
 * @param id - Dictionary ID (e.g., 'python-v1', 'typescript-v1')
 * @returns Promise resolving to the static dictionary
 *
 * @example
 * ```typescript
 * import { loadStaticDictionary, compress } from '@delta-ltsc/sdk';
 *
 * const pythonDict = await loadStaticDictionary('python-v1');
 *
 * const result = await compress(tokens, {
 *   staticDictionary: pythonDict,
 * });
 * ```
 */
export async function loadStaticDictionary(
  id: StaticDictionaryId
): Promise<StaticDictionary> {
  const data = STATIC_DICTIONARIES[id];
  if (!data) {
    throw new Error(`Unknown static dictionary: ${id}`);
  }

  return parseDictionaryJson(data as DictionaryJson);
}

/**
 * Parse dictionary JSON into StaticDictionary.
 */
function parseDictionaryJson(data: DictionaryJson): StaticDictionary {
  const entries = new Map<number, readonly number[]>();
  const patterns = new Map<string, number>();

  for (const entry of data.entries) {
    entries.set(entry.metaToken, entry.pattern);
    patterns.set(JSON.stringify(entry.pattern), entry.metaToken);
  }

  return {
    id: data.id,
    version: data.version,
    name: data.name,
    description: data.description,
    entries,
    patterns,
  };
}

/**
 * Create a custom static dictionary from patterns.
 *
 * @param id - Unique identifier for the dictionary
 * @param patterns - Array of token patterns to include
 * @param startMetaToken - Starting meta-token ID (default: 0xFFFF8000)
 * @returns StaticDictionary ready for use
 */
export function createStaticDictionary(
  id: string,
  patterns: number[][],
  startMetaToken = 0xffff8000
): StaticDictionary {
  const entries = new Map<number, readonly number[]>();
  const patternMap = new Map<string, number>();

  for (let i = 0; i < patterns.length; i++) {
    const pattern = patterns[i];
    const metaToken = startMetaToken + i;
    entries.set(metaToken, pattern);
    patternMap.set(JSON.stringify(pattern), metaToken);
  }

  return {
    id,
    version: '1.0.0',
    name: id,
    description: `Custom dictionary: ${id}`,
    entries,
    patterns: patternMap,
  };
}

/**
 * List available built-in dictionaries.
 */
export function listStaticDictionaries(): StaticDictionaryId[] {
  return Object.keys(STATIC_DICTIONARIES) as StaticDictionaryId[];
}

/**
 * Check if a dictionary ID is a built-in dictionary.
 */
export function isBuiltinDictionary(id: string): id is StaticDictionaryId {
  return id in STATIC_DICTIONARIES;
}
