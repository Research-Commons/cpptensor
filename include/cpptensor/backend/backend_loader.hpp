#pragma once

namespace cpptensor {
    /**
     * Register backend kernels.
     *
     * Public ops now trigger this lazily on first use, so calling this
     * manually is optional. The function is idempotent and can still be used
     * to warm the registry explicitly during startup.
     */
    void initialize_kernels();
}
