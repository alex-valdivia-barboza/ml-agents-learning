// Recuadro "Necesitas antes / Es requisito de" al final de cada fundamento.
// Este archivo es la PLANTILLA: requisitos.js se genera con `python generar_requisitos.py`,
// que reemplaza DATOS por los requisitos leídos del contenido de cada página.
(function () {
  var DATOS = /*DATOS*/null;
  if (!DATOS) return;
  // La vista "Requisitos 2D" de index.html usa los mismos datos
  window.REQUISITOS = DATOS;

  var actual = decodeURIComponent(location.pathname.split('/').pop());
  var yo = DATOS.fundamentos[actual];
  if (!yo) return;

  var posicion = {};
  DATOS.orden.forEach(function (archivo, i) { posicion[archivo] = i; });
  var necesita = yo.r.slice();
  var esRequisitoDe = DATOS.orden.filter(function (archivo) {
    return DATOS.fundamentos[archivo].r.indexOf(actual) !== -1;
  });

  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; });
  }

  // Agrupa por lección, en orden de estudio
  function grupos(archivos) {
    archivos.sort(function (a, b) { return posicion[a] - posicion[b]; });
    var html = '', leccion = null;
    archivos.forEach(function (archivo) {
      var f = DATOS.fundamentos[archivo];
      if (f.l !== leccion) {
        if (leccion !== null) html += '</ul>';
        leccion = f.l;
        var titulo = f.l === yo.l ? 'En esta lección' : 'Lección ' + f.l + ' · ' + DATOS.lecciones[f.l];
        html += '<p class="req-leccion">' + esc(titulo) + '</p><ul>';
      }
      html += '<li><a href="' + esc(archivo) + '">' + esc(f.t) + '</a></li>';
    });
    return html + (leccion !== null ? '</ul>' : '');
  }

  var estilo = document.createElement('style');
  estilo.textContent =
    '.req-box{margin:40px 0 0;padding:18px 20px;border:1px solid var(--border);border-radius:12px;background:var(--bg-card)}' +
    '.req-box h2{margin:0 0 4px;font-size:1.05rem}' +
    '.req-box .req-intro{margin:0 0 14px;font-size:.85rem;color:var(--text-dim)}' +
    '.req-cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:18px}' +
    '.req-col h3{margin:0 0 6px;font-size:.9rem;color:var(--req-acento)}' +
    '.req-leccion{margin:10px 0 2px;font-size:.75rem;font-weight:700;color:var(--text-dim)}' +
    '.req-col ul{margin:0;padding-left:18px;font-size:.88rem}' +
    '.req-col li{margin:2px 0}' +
    '.req-col a{color:var(--text);text-decoration:none;border-bottom:1px dotted var(--text-dim)}' +
    '.req-col a:hover{color:var(--req-acento)}' +
    '.req-vacio{margin:0;font-size:.85rem;color:var(--text-dim)}';
  document.head.appendChild(estilo);

  var caja = document.createElement('section');
  caja.className = 'req-box';
  caja.setAttribute('aria-label', 'Cómo se conecta este fundamento');
  var tag = document.querySelector('.module-tag');
  if (tag) caja.style.setProperty('--req-acento', getComputedStyle(tag).color);
  caja.innerHTML =
    '<h2>Cómo se conecta</h2>' +
    '<p class="req-intro">Fundamentos que esta página cita: los anteriores son lo que conviene tener claro primero, y los posteriores son los que se apoyan en este.</p>' +
    '<div class="req-cols">' +
      '<div class="req-col"><h3>Necesitas antes</h3>' +
        (necesita.length ? grupos(necesita) : '<p class="req-vacio">No se apoya en otro fundamento: puedes empezar por aquí.</p>') +
      '</div>' +
      '<div class="req-col"><h3>Es requisito de</h3>' +
        (esRequisitoDe.length ? grupos(esRequisitoDe) : '<p class="req-vacio">Todavía ningún fundamento posterior lo usa.</p>') +
      '</div>' +
    '</div>';

  var pie = document.querySelector('footer');
  if (pie) pie.parentNode.insertBefore(caja, pie);
  else (document.querySelector('.wrap') || document.body).appendChild(caja);
})();
