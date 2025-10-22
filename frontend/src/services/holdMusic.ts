/**
 * Hold Music Service
 *
 * Generates and plays looping hold music while waiting for AI response.
 */

class HoldMusicService {
  private audioContext: AudioContext | null = null;
  private gainNode: GainNode | null = null;
  private isPlaying: boolean = false;
  private loopTimeout: number | null = null;
  private volume: number = 0.15; // 15% volume (low)

  /**
   * Initialize audio context (lazy initialization)
   */
  private initAudioContext() {
    if (!this.audioContext) {
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      this.gainNode = this.audioContext.createGain();
      this.gainNode.gain.value = this.volume;
      this.gainNode.connect(this.audioContext.destination);
    }
  }

  /**
   * Generate a beep tone
   */
  private generateBeep(frequency: number, duration: number, startTime: number): void {
    if (!this.audioContext || !this.gainNode) return;

    const oscillator = this.audioContext.createOscillator();
    const beepGain = this.audioContext.createGain();

    oscillator.type = 'sine';
    oscillator.frequency.value = frequency;

    // Apply fade in/out to avoid clicks
    const fadeTime = 0.01;
    beepGain.gain.setValueAtTime(0, startTime);
    beepGain.gain.linearRampToValueAtTime(1, startTime + fadeTime);
    beepGain.gain.setValueAtTime(1, startTime + duration - fadeTime);
    beepGain.gain.linearRampToValueAtTime(0, startTime + duration);

    oscillator.connect(beepGain);
    beepGain.connect(this.gainNode);

    oscillator.start(startTime);
    oscillator.stop(startTime + duration);
  }

  /**
   * Play the rapid double-beep pattern
   */
  private playPattern() {
    if (!this.audioContext || !this.gainNode || !this.isPlaying) return;

    const now = this.audioContext.currentTime;
    const beepDuration = 0.08; // 80ms
    const shortPause = 0.08;   // 80ms between beeps
    const longPause = 1.5;     // 1.5s before repeating

    // Play first beep
    this.generateBeep(750, beepDuration, now);

    // Play second beep
    this.generateBeep(750, beepDuration, now + beepDuration + shortPause);

    // Schedule next pattern
    const patternDuration = (beepDuration * 2 + shortPause + longPause) * 1000;
    this.loopTimeout = window.setTimeout(() => {
      if (this.isPlaying) {
        this.playPattern();
      }
    }, patternDuration);
  }

  /**
   * Start playing hold music
   */
  start() {
    if (this.isPlaying) return;

    this.initAudioContext();
    this.isPlaying = true;

    // Resume audio context if it was suspended (browser autoplay policy)
    if (this.audioContext?.state === 'suspended') {
      this.audioContext.resume();
    }

    this.playPattern();
  }

  /**
   * Stop playing hold music
   */
  stop() {
    this.isPlaying = false;

    if (this.loopTimeout !== null) {
      clearTimeout(this.loopTimeout);
      this.loopTimeout = null;
    }

    // Note: We don't close the audio context to allow reuse
  }

  /**
   * Play a single beep (used after assistant finishes speaking)
   */
  playSingleBeep() {
    this.initAudioContext();

    // Resume audio context if it was suspended
    if (this.audioContext?.state === 'suspended') {
      this.audioContext.resume();
    }

    if (!this.audioContext || !this.gainNode) return;

    const now = this.audioContext.currentTime;
    const beepDuration = 0.08; // 80ms, same as hold music

    // Play single beep with same frequency as hold music
    this.generateBeep(750, beepDuration, now);
  }

  /**
   * Check if hold music is currently playing
   */
  getIsPlaying(): boolean {
    return this.isPlaying;
  }

  /**
   * Set volume (0.0 to 1.0)
   */
  setVolume(volume: number) {
    this.volume = Math.max(0, Math.min(1, volume));
    if (this.gainNode) {
      this.gainNode.gain.value = this.volume;
    }
  }

  /**
   * Cleanup resources
   */
  dispose() {
    this.stop();
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
      this.gainNode = null;
    }
  }
}

// Export singleton instance
export const holdMusicService = new HoldMusicService();
