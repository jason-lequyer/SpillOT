plugins {
    // Optional shadow/fat jar support (no extra dependencies are bundled at present)
    id("com.gradleup.shadow") version "8.3.5"
    // QuPath Gradle extension convention plugin
    id("qupath-conventions")
}

qupathExtension {
    name = "qupath-extension-spillot"
    group = "qupath.ext.spillot"
    version = "0.1.0"
    description = "SpillOT: user-guided spillover removal for highly multiplexed images"
    automaticModule = "qupath.ext.spillot"
}

dependencies {
    shadow(libs.bundles.qupath)
    shadow(libs.bundles.logging)
    shadow(libs.qupath.fxtras)
}

// Bundle the same Python backend that the Fiji plugin and the command line use,
// so there is a single source of truth for the algorithm.
tasks.processResources {
    from("../plugins/SpillOT/SpillOT.py") {
        into("qupath/ext/spillot")
    }
}
