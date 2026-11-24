plugins {
    // Optional shadow/fat jar support (no extra dependencies are bundled at present)
    id("com.gradleup.shadow") version "8.3.5"
    // QuPath Gradle extension convention plugin
    id("qupath-conventions")
}

qupathExtension {
    name = "qupath-extension-spillot"
    group = "qupath.ext.spillot"
    version = "0.2.3"
    description = "SpillOT: user-guided spillover removal for highly multiplexed images"
    automaticModule = "qupath.ext.spillot"
}

dependencies {
    shadow(libs.bundles.qupath)
    shadow(libs.bundles.logging)
    shadow(libs.qupath.fxtras)
    testImplementation(libs.bundles.qupath)
    testImplementation(libs.bundles.logging)
    testImplementation(libs.qupath.fxtras)
    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher:1.11.4")
}

tasks.test {
    useJUnitPlatform()
}

// Bundle the same Python backend that the Fiji plugin and the command line use,
// so there is a single source of truth for the algorithm.
tasks.processResources {
    from(listOf("../plugins/SpillOT/SpillOT.py", "../plugins/SpillOT/SpillOTMemory.py")) {
        into("qupath/ext/spillot")
    }
}
