import { describe, expect, it, vi } from 'vitest'

vi.mock('@/hermes', () => ({
  getHermesConfigRecord: vi.fn(async () => ({})),
  saveHermesConfig: vi.fn(async () => undefined)
}))

import {
  $voiceSilenceMs,
  $voiceStopPhrase,
  applyVoiceSilenceFromConfig,
  applyVoiceStopPhraseFromConfig
} from './voice-prefs'

describe('applyVoiceSilenceFromConfig', () => {
  it('converts the configured seconds to milliseconds', () => {
    applyVoiceSilenceFromConfig({ voice: { silence_duration: 0.85 } })
    expect($voiceSilenceMs.get()).toBe(850)
  })

  it('keeps a safe floor and preserves the desktop default for invalid values', () => {
    applyVoiceSilenceFromConfig({ voice: { silence_duration: 0.1 } })
    expect($voiceSilenceMs.get()).toBe(300)

    applyVoiceSilenceFromConfig({ voice: { silence_duration: 'fast' } })
    expect($voiceSilenceMs.get()).toBe(1_250)

    applyVoiceSilenceFromConfig({ voice: { silence_duration: Number.MAX_VALUE } })
    expect($voiceSilenceMs.get()).toBe(1_250)
  })
})

describe('applyVoiceStopPhraseFromConfig', () => {
  it('defaults to "stop" when the key is absent (backend default applies)', () => {
    applyVoiceStopPhraseFromConfig({ voice: {} })
    expect($voiceStopPhrase.get()).toBe('stop')

    applyVoiceStopPhraseFromConfig(null)
    expect($voiceStopPhrase.get()).toBe('stop')
  })

  it('uses the first configured phrase so a custom phrase renders correctly', () => {
    applyVoiceStopPhraseFromConfig({ voice: { stop_phrases: ['goodbye hermes', 'stop'] } })
    expect($voiceStopPhrase.get()).toBe('goodbye hermes')
  })

  it('coerces a bare string like the backend does', () => {
    applyVoiceStopPhraseFromConfig({ voice: { stop_phrases: 'halt' } })
    expect($voiceStopPhrase.get()).toBe('halt')
  })

  it('null phrase when stop phrases are disabled — no notice is shown', () => {
    applyVoiceStopPhraseFromConfig({ voice: { stop_phrases: [] } })
    expect($voiceStopPhrase.get()).toBeNull()
  })

  it('malformed entries are skipped; all-blank list disables', () => {
    applyVoiceStopPhraseFromConfig({ voice: { stop_phrases: ['  ', ''] } })
    expect($voiceStopPhrase.get()).toBeNull()
  })
})
