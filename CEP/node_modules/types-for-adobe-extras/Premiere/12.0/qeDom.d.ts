// ----------------- DISCLAIMER -------------------
// The Premiere Pro QE DOM is offically UNSUPPORTED/
// This means these methods can change at any time
// and usage will not be supported by the Adobe team.
//
// Use at your own risk.
//
// --------------- END DISCLAIMER ------------------
//
// Type definitions for Premiere Pro's QE DOM API
// Definitions based off of Eric Robinson's work

/**
 * The `qe` object provides access to the application's
 * "Quality Engineering" DOM APIs.
 *
 * **WARNING:** This API is meant for Adobe-internal use
 * only. It is subject to change at any time and may
 * contain serious bugs. Use these APIs at your own peril.
 * **YOU HAVE BEEN WARNED.**
 */

interface QEApplication {
  ea: QEEA
  name: string
  version: string
  config: string
  location: string
  platform: string
  language: string
  /** The QE representation of the Project. */
  project: QEProject
  codeProfiler: CodeProfiler
  audioChannelMapping: number
  source: QESource
  getDebugDatabaseEntry(): any
  executeConsoleCommand(): any
  outputToConsole(): any
  exit(): any
  stop(): any
  open(): any
  enablePlayStats(): any
  startPlayback(): any
  stopPlayback(): any
  newProject(): any
  wait(): any
  localize(): any
  enablePerformanceLogging(): any
  disablePerformanceLogging(): any
  isPerformanceLoggingEnabled(): any
  getSequencePresets(): any
  setAudioChannelMapping(): any
  beginDroppedFrameLogging(): any
  log(): any
  getModalWindowID(): any
  setDebugDatabaseEntry(): any
}
interface QEEA {
  getSessionSyncStatus(): any
  getRemoteServerBuildVersion(): any
  isSyncCommandEnabled(): any
  isShareCommandEnabled(): any
  isLoggedIn(): any
  convertProjectIntoProduction(): any
  convertProductionIntoProject(): any
  isConvertProjectIntoProductionRunning(): any
  getProcessID(): any
  getUsername(): any
  getAuthToken(): any
  setAuthToken(): any
  getProductionList(): any
  getArchivedProductionList(): any
  getInviteList(): any
  openProduction(): any
  sync(): any
  canShare(): any
  share(): any
  getConflicts(): any
  resolveConflict(): any
  doesEditingSessionHaveLocalMedia(): any
  benchmarkReflectEverything(): any
  waitForCurrentReflectionToComplete(): any
  isCollaborationOnly(): any
  isHostedCollaborationOnly(): any
  getLoggedInDataServerVersion(): any
  getDiscoveryURL(): any
  getURL(): any
  postURL(): any
  createProduction(): any
  setLocalHubConnectionStatus(): any
  getLoggedInUserDisplayId(): any
  renameProduction(): any
}
interface QEProject {
  readonly currentRendererName: string
  readonly importFailures: any[]
  readonly isAudioConforming: boolean
  readonly isAudioPeakGenerating: boolean
  readonly isIndexing: boolean
  readonly name: string
  readonly numActiveProgressItems: number
  readonly numAudioPeakGeneratedFiles: number
  readonly numBins: number
  readonly numConformedFiles: number
  readonly numIndexedFiles: number
  readonly numItems: number
  readonly numSequenceItems: number
  readonly numSequences: number
  readonly path: string
  /**
   * Create new Black Video
   *
   * @param width width in pixels
   * @param height height in pixels
   * @param framerate framerate
   * @param aspectNumerator pixel aspect numerator
   * @param aspectDenominator pixel aspect denominator
   *
   * @example qe.project.newBlackVideo(1920, 1080, 24, 1, 1)
   */
  newBlackVideo(
    width: number,
    height: number,
    framerate: number,
    aspectNumerator: number,
    aspectDenominator: number,
  ): boolean
  /** Get the QE representation of the currently active Sequence.
   *
   * @example qe.project.getActiveSequence();
   */
  getActiveSequence(): Sequence
  /**
   *
   * @param name The name of the new Sequence.
   * @param pathToPreset The path the the sequence preset to use.
   *
   * @example qe.project.newSequence('Rough Cut', 'path/to/preset.sqpreset');
   *
   * See: https://github.com/Adobe-CEP/Samples/blob/00366bf8a86e44bd83704a04a8f4f0cdc75fd38f/PProPanel/jsx/PPRO/Premiere.jsx#L425
   */
  newSequence(name: string, pathToPreset: string): boolean
  /**
   * Delete cached preview files for the specified media type.
   * @example qe.project.deletePreviewFiles("FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF"); // any
   */
  deletePreviewFiles(type: MediaType): boolean

  findItemByID(pp0: string): object
  flushCache(): boolean
  getAudioEffectByName(pp0: string, pchannelType: number, pp2: boolean): object
  getAudioEffectList(peffectType: number, pp1: boolean): string[]
  getAudioTransitionByName(pp0: string, pp1: boolean): object
  getAudioTransitionList(pp0: boolean): string[]
  getBinAt(pp0: number): QEProjectItemContainer
  getItemAt(pp0: number): object
  getRemainingMetadataCacheIndexCount(): number
  getRendererNames(): string[]
  getSequenceAt(pp0: number): Sequence
  getSequenceItemAt(pp0: number): SequenceItem
  /**
   * Gets the effect by name
   * @param name The name of the Effect
   * @param pp1 ?
   * @returns a VideoEffect object to be used with QETrackItem.addVideoEffect()
   *
   * @example qe.project.getVideoEffectByName("Lumetri Color");
   */
  getVideoEffectByName(name: string, pp1?: boolean): VideoEffect
  getVideoEffectList(peffectType: number, pp1: boolean): string[]
  getVideoTransitionByName(pp0: string, pp1: boolean): object
  getVideoTransitionList(pp0: number, pp1: boolean): string[]
  (pp0: any[], pisNumberedStills: boolean): boolean
  importAEComps(pp0: string, pp1: any[]): boolean
  importAllAEComps(pp0: string): boolean
  importFiles(pp0: any[], pp1: boolean, pp2: boolean): boolean
  (pp0: number, pp1: number, pp2: number, pp3: number, pp4: number, pp5: number): boolean
  newBin(pp0: string): QEProjectItemContainer
  newCaption(
    pstandard: string,
    pstream: string,
    ptimebase1: string,
    pgrid: string,
    ptimebase2: string,
    pparnum: number,
    pparden: number,
    ptestData: boolean,
  ): boolean
  (pp0: number, pp1: number, pp2: number, pp3: number, pp4: number): boolean
  newSmartBin(pp0: string, pp1: string): boolean
  newTransparentVideo(pp0: number, pp1: number, pp2: number, pp3: number, pp4: number): boolean
  newUniversalCountingLeader(
    pp0: number,
    pp1: number,
    pp2: number,
    pp3: number,
    pp4: number,
    pp5: number,
  ): boolean
  redo(): boolean
  resetNumFilesCounter(): boolean
  save(): boolean
  (pp0: string): boolean
  setRenderer(pp0: string): boolean
  sizeOnDisk(): number
  undo(): boolean
  undoStackIndex(): number
  // undoStack(): any // no longer available?
}
interface CodeProfiler {
  start(): any
  stop(): any
  reset(): any
  get(): any
}
interface QEProjectItemContainer {
  readonly name: string
  readonly numBins: number
  readonly numItems: number
  readonly numSequenceItems: number
  readonly numSequences: number
  flushCache(): boolean
  getBinAt(pp0: number): QEProjectItemContainer
  getItemAt(pp0: number): QEProjectItem
  getSequenceAt(pp0: number): object
  getSequenceItemAt(pp0: number): object
  newBin(pp0: string): boolean
}
interface QEProjectItem {
  readonly clip: QEMasterClip
  readonly filePath: string
  readonly mediaSyncStatus: string
  readonly name: string
  automateToSequence(pp0: object, pp1: number, pp2: number, pp3: number, pp4: number): boolean
  containsSpeechTrack(): boolean
  createProxy(pp0: string, pp1: string): boolean
  getMetadataSize(): number
  isAudioConforming(): boolean
  isAudioPeakGenerating(): boolean
  isIndexing(): boolean
  isOffline(): boolean
  isPending(): boolean
  linkMedia(pp0: string, pp1: boolean): boolean
  openInSource(): boolean
  rename(passetName: string): boolean
  setOffline(): boolean
}
interface VideoEffect {
  name: string
}
interface QETrackItem {
  readonly alignment: number
  readonly antiAliasQuality: number
  readonly borderColor: number
  readonly borderWidth: number
  readonly duration: number
  readonly end: number
  readonly frameBlend: number
  readonly mediaType: number
  readonly multicamEnabled: number
  readonly name: number
  readonly numComponents: number
  readonly reverse: number
  readonly reversed: number
  readonly scaleToFrameSize: number
  readonly speed: number
  readonly start: number
  readonly staticClipGain: number
  readonly switchSources: number
  readonly timeInterpolationType: number
  readonly type: number
  // readonly startPercent: number // Don't Use, will crash Premiere
  // readonly endPercent: number // Don't Use, will crash Premiere

  addAudioEffect(p0: object): boolean
  addTransition(
    p0: object,
    p1: boolean,
    p2: string,
    p3: string,
    p4: number,
    p5: boolean,
    p6: boolean,
  ): boolean
  /**
   * Add a video effect to a QETrackItem
   *
   * @param effect The name of the new Sequence.
   * @example item.addVideoEffect(qe.project.getVideoEffectByName("Lumetri Color"));
   */
  addVideoEffect(effect: VideoEffect): boolean
  canDoMulticam(): boolean
  getClipPanComponent(): object
  getComponentAt(p0: number): object
  getProjectItem(): object
  move(
    p0: string,
    p1: boolean,
    p2: boolean,
    p3: boolean,
    p4: boolean,
    p5: boolean,
    p6: boolean,
  ): boolean
  moveToTrack(p0: number, p1: number, p2: string, p3: boolean): boolean
  remove(p0: boolean, p1: boolean): boolean
  removeEffects(p0: boolean, p1: boolean, p2: boolean, p3: boolean, p4: boolean): boolean
  rippleDelete(): boolean
  roll(p0: string, p1: boolean, p2: boolean): boolean
  setAntiAliasQuality(p0: number): boolean
  setBorderColor(p0: number, p1: number, p2: number): boolean
  setBorderWidth(p0: number): boolean
  setEndPercent(p0: number): boolean
  setEndPosition(p0: number, p1: number): boolean
  setFrameBlend(p0: boolean): boolean
  setMulticam(p0: boolean): boolean
  setName(p0: string): boolean
  setReverse(p0: boolean): boolean
  setScaleToFrameSize(p0: boolean): boolean
  setSpeed(p0: number, p1: string, p2: boolean, p3: boolean, p4: boolean): boolean
  setStartPercent(p0: number): boolean
  setStartPosition(p0: number, p1: number): boolean
  setSwitchSources(p0: boolean): boolean
  setTimeInterpolationType(p0: number): boolean
  slide(p0: string, p1: boolean): boolean
  slip(p0: string, p1: boolean): boolean
}
interface SequenceItem {
  name: string
  openInSource(): any
  isPending(): boolean
  isAudioConforming(): boolean
  isIndexing(): boolean
  isAudioPeakGenerating(): boolean
}
interface QESource {
  player: QEPlayer
  clip: QEMasterClip
  openFilePath(): any
}
interface QEPlayer {
  loopPlayback: false
  droppedFrames: string
  totalFrames: string
  audioDropouts: number
  audioMediaNotFound: number
  audioClockJitters: number
  audioIOOverloads: number
  audioIODropouts: number
  audioPrefetchBehinds: number
  audioDeviceLoadAvg: number
  audioDeviceLoadMin: number
  audioDeviceLoadMax: number
  audioDeviceLoadStdDev: number
  play(): any
  stop(): any
  step(): any
  startScrubbing(): any
  setLoopPlayback(): any
  enableStatistics(): any
  disableStatistics(): any
  clearAudioDropoutStatus(): any
  captureAudioDeviceLoad(): any
  getPosition(): any
}
interface QEMasterClip {
  readonly audioChannelType: number
  readonly audioFrameRate: number
  readonly audioNumChannels: number
  readonly audioSampleSize: number
  readonly duration: string
  readonly filePath: string
  readonly name: string
  readonly videoFieldType: number
  readonly videoFrameHeight: number
  readonly videoFrameRate: number
  readonly videoFrameWidth: number
  readonly videoHasAlpha: boolean
  readonly videoPixelAspectRatio: string
  clearChildClips(): boolean
  clearInPoint(): boolean
  clearOutPoint(): boolean
  containsCaptioningStream(pp0: string, pp1: string): boolean
  containsCaptions(): boolean
  getCaptioningStreamAt(pp0: number): object
  hasChildClipsInUse(): boolean
  numCaptioningStreams(): number
  numOfChildClips(): number
  numOfChildClipsInUse(): number
  setAudioInPoint(pp0: string): boolean
  setAudioOutPoint(pp0: string): boolean
  setDuration(pp0: string): boolean
  setInPoint(pp0: string): boolean
  setOutPoint(pp0: string): boolean
  setVideoInPoint(pp0: string): boolean
  setVideoOutPoint(pp0: string): boolean
}
interface Sequence {
  /** Name of the sequence. */
  name: string
  numVideoTracks: 3
  numAudioTracks: 4
  /**
   * The Current Time Indicator for the active sequence.
   *
   * See: https://github.com/Adobe-CEP/Samples/blob/00366bf8a86e44bd83704a04a8f4f0cdc75fd38f/PProPanel/jsx/PPRO/Premiere.jsx#L499
   */
  CTI: Time
  inPoint: Time
  outPoint: Time
  workInPoint: Time
  workOutPoint: Time
  useMaxBitDepth: boolean
  useMaxRenderQuality: boolean
  videoDisplayFormat: 110
  audioDisplayFormat: 200
  previewFrameSize: [number, number]
  presetList: string[]
  previewPresetPath: string
  previewPresetCodec: number
  editingMode: string
  videoFrameSize: [number, number]
  audioFrameRate: number
  videoFrameRate: number
  par: number
  fieldType: number
  guid: string
  player: QEPlayer
  multicam: Multicam
  getVideoTrackAt(idx: number): Track
  getAudioTrackAt(idx: number): Track
  makeCurrent(): any
  close(): any
  isOpen(): boolean
  renderPreview(): any
  renderAll(): any
  renderAudio(): any
  /**
   * Adds tracks to the current sequence as specified.
   * @param numVideo The number of Video Tracks to add. [Default: 1]
   * @param videoIndex The index at which added Video Track(s) should be inserted into the Video Tracks list. [Default: 0]
   * @param numAudio The number of Audio Tracks to add. [Default: 1]
   * @param audioChannelType The Audio Channel configuration that added Audio Tracks should support. [Default: 1]
   * @param audioIndex The index at which added Audio Track(s) should be inserted into the Audio Tracks list. [Default: 0]
   * @param numSubAudio The number of SubAudio Tracks (Submix Tracks) to add. [Default: 0]
   * @param subAudioChannelType The Audio Channel configuration that added SubAudio Tracks (Submix tracks) should support. [Default: 1]
   * @param subAudioIndex The index at which added SubAudio Track(s) (Submix Track(s)) should be inserted into the SubAudio Tracks list. [Default: 0?]
   */
  addTracks(
    numVideo?: number,
    videoIndex?: number,
    numAudio?: number,
    audioChannelType?: AudioChannelType,
    audioIndex?: number,
    numSubAudio?: number,
    subAudioChannelType?: AudioChannelType,
    subAudioIndex?: number,
  ): boolean
  removeTracks(): any
  removeVideoTrack(): any
  removeAudioTrack(): any
  removeEmptyVideoTracks(): any
  removeEmptyAudioTracks(): any
  exportToAME(): any
  exportDirect(): any
  getExportComplete(): any
  createExportJob(): any
  /**
   * Retrieve the file
   */
  getExportFileExtension(): any
  razor(): any
  setCTI(): any
  setInPoint(): any
  setOutPoint(): any
  setInOutPoints(): any
  lift(): any
  extract(): any
  setWorkInPoint(): any
  setWorkOutPoint(): any
  setWorkInOutPoints(): any
  lockTracks(): any
  syncLockTracks(): any
  muteTracks(): any
  deletePreviewFiles(): any
  getRedBarTimes(): any
  getGreenBarTimes(): any
  getYellowBarTimes(): any
  getEmptyBarTimes(): any
  setUseMaxBitDepth(): any
  setUseMaxRenderQuality(): any
  setVideoDisplayFormat(): any
  setAudioDisplayFormat(): any
  setPreviewFrameSize(): any
  setPreviewPresetPath(): any
  exportFrameDPX(): any
  /**
   * See: https://github.com/Adobe-CEP/Samples/blob/00366bf8a86e44bd83704a04a8f4f0cdc75fd38f/PProPanel/jsx/PPRO/Premiere.jsx#L71
   *
   * @param timecode The timecode of the frame to export. Format may require replacing [semi-]colons (";:") with underscores ("_").
   * @param filePath The path (including filename) at which to export the png file.
   */
  exportFramePNG(timecode: string, filePath: string): any
  exportFrameTarga(): any
  exportFrameJPEG(): any
  exportFrameTIFF(): any
  flushCache(): any
}
interface Time {
  ticks: number
  secs: number
  frames: number
  timecode: string
}
interface Multicam {
  play(): any
  stop(): any
  changeCamera(): any
  record(): any
  enable(): any
}
interface Track {
  name: string
  index: number
  type: string
  numItems: number
  numTransitions: number
  numComponents: number
  /**
   * index of clip along track, including gaps
   */
  getItemAt(index: number): QETrackItem
  getTransitionAt(): any
  setName(): any
  insert(): any
  overwrite(): any
  addAudioEffect(): any
  getComponentAt(): any
  razor(): any
  setLock(): any
  isLocked(): boolean
  setSyncLock(): any
  isSyncLocked(): boolean
  setMute(): any
  isMuted(): boolean
}
/**
 * Options are:
 * - `0`: Mono
 * - `1`: Stero
 * - `2`: 5.1
 * - `3`: Multichannel
 * - `4`: 4 Channel
 * - `5`: 8 Channel
 *
 */
type AudioChannelType = 0 | 1 | 2 | 3 | 4 | 5
/**
 * Options are:
 * - `228CDA18-3625-4d2d-951E-348879E4ED93`: Video
 * - `80B8E3D5-6DCA-4195-AEFB-CB5F407AB009`: Audio
 * - `FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF`: Any
 */
type MediaType =
  | "228CDA18-3625-4d2d-951E-348879E4ED93"
  | "80B8E3D5-6DCA-4195-AEFB-CB5F407AB009"
  | "FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF"

declare const qe: undefined | QEApplication
